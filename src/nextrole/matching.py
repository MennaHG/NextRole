"""RAG-based semantic job matching with an ephemeral Chroma index."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from .config import Settings
from .models import (
    CandidateProfile,
    EvaluationDecision,
    EvidenceDebugTrace,
    JobEvaluation,
    JobPosting,
    MatchingReport,
    RequirementAssessment,
)


class MatchingError(RuntimeError):
    """A matching dependency or evaluation failed."""


class Retriever(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]:
        """Return the most relevant career-history documents."""


class DeterministicEmbeddings(Embeddings):
    """Offline embeddings used only by fixture mode and tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:16]]


class CareerIndex:
    """A single-run vector index over career-history chunks."""

    def __init__(self, retriever: Retriever, document_count: int) -> None:
        self._retriever = retriever
        self.document_count = document_count

    def retrieve(self, query: str, k: int) -> list[Document]:
        return self._retriever.similarity_search(query, k=k)


def chunk_career_history(
    profile: CandidateProfile, *, chunk_size: int = 1200, chunk_overlap: int = 120
) -> list[Document]:
    """Split only career history; preferences stay outside the vector index."""
    if chunk_overlap >= chunk_size:
        raise MatchingError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except (
        ImportError
    ):  # pragma: no cover - used only in dependency-limited offline runs
        return _fallback_chunks(profile.career_history, chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(profile.career_history)
    return [
        Document(
            page_content=chunk,
            metadata={"source": "career_history", "chunk_index": index},
        )
        for index, chunk in enumerate(chunks)
    ]


def build_career_index(
    profile: CandidateProfile,
    embeddings: Embeddings,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
    vector_store_factory: Callable[..., Any] | None = None,
) -> CareerIndex:
    """Build one in-memory Chroma collection for the complete matching run."""
    documents = chunk_career_history(
        profile, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if vector_store_factory is None:
        try:
            from langchain_chroma import Chroma
        except (
            ImportError
        ):  # pragma: no cover - used only in dependency-limited offline runs
            vector_store_factory = _InMemoryVectorStore
        else:
            vector_store_factory = Chroma
    store = vector_store_factory(
        collection_name=f"nextrole_{uuid4().hex}",
        embedding_function=embeddings,
        persist_directory=None,
    )
    store.add_documents(documents)
    return CareerIndex(store, len(documents))


def match_jobs(
    profile: CandidateProfile,
    jobs: Sequence[JobPosting],
    settings: Settings,
    *,
    live: bool = False,
    chat_model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    index: CareerIndex | None = None,
    debug_evidence: bool = False,
) -> MatchingReport:
    """Evaluate jobs with one structured model call per job."""
    unique_jobs, duplicates = _deduplicate_jobs(jobs)
    eligible_jobs: list[JobPosting] = []
    skipped_jobs: list[dict[str, str]] = []
    candidate_evidence_available = bool(profile.career_history.strip())
    for job in unique_jobs:
        if job.validity_status == "invalid":
            skipped_jobs.append({"title": job.title, "reason": "confirmed invalid"})
        elif not _has_job_evidence(job):
            skipped_jobs.append(
                {
                    "title": job.title,
                    "reason": "description and requirements are missing",
                }
            )
        elif not candidate_evidence_available:
            skipped_jobs.append(
                {
                    "title": job.title,
                    "reason": "candidate career evidence is unavailable",
                }
            )
        else:
            eligible_jobs.append(job)
    selected = eligible_jobs[: settings.match_max_jobs]
    if not live:
        embeddings = embeddings or DeterministicEmbeddings()
        chat_model = chat_model or FixtureChatModel()
        mode = "fixture"
    else:
        if chat_model is None or embeddings is None:
            raise MatchingError(
                "Live matching requires configured chat and embedding providers."
            )
        mode = "live"
    if not selected:
        return MatchingReport(
            jobs_received=len(jobs),
            duplicates_removed=duplicates,
            mode=mode,
            evidence_debug=[] if debug_evidence else None,
            skipped_jobs=skipped_jobs,
        )
    index = index or build_career_index(
        profile,
        embeddings,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    report = MatchingReport(
        jobs_received=len(jobs),
        duplicates_removed=duplicates,
        embedding_documents=index.document_count,
        mode=mode,
        evidence_debug=[] if debug_evidence else None,
        skipped_jobs=skipped_jobs,
    )
    if not selected:
        return report
    structured = chat_model.with_structured_output(EvaluationDecision)
    for job in selected:
        context = index.retrieve(_job_query(job), k=settings.rag_top_k)
        prompt = _prompt(profile, job, context)
        report.llm_calls_attempted += 1
        try:
            response = structured.invoke(prompt)
            evaluation = _coerce_evaluation(response, job)
            usage = _extract_token_usage(response)
            if usage:
                report.token_usage = _merge_usage(report.token_usage, usage)
        except (ValidationError, TypeError, ValueError, MatchingError) as error:
            evaluation = _uncertain_evaluation(
                job, f"Evaluation failed or was malformed: {error}"
            )
        report.evaluations.append(evaluation)
        if debug_evidence:
            report.evidence_debug.append(
                EvidenceDebugTrace(
                    job_title=job.title,
                    retrieved_chunks=[document.page_content for document in context],
                    job_requirements=job.requirements,
                    requirement_assessments=evaluation.requirement_assessments,
                    eligibility=evaluation.eligibility,
                    demonstrated_fit=evaluation.demonstrated_fit,
                    evidence_coverage=evaluation.evidence_coverage,
                    match_score=evaluation.match_score,
                )
            )
        report.jobs_evaluated += 1
    report.evaluations.sort(
        key=lambda item: (
            {"eligible": 0, "uncertain": 1, "ineligible": 2}[item.eligibility],
            -item.match_score,
        )
    )
    report.embedding_operations = 1
    return report


def load_jobs_json(path: Path) -> list[JobPosting]:
    """Load a JSON list, discovery report, or ``{"jobs": [...]}`` input."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MatchingError(f"Jobs input file not found: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise MatchingError(f"Could not read jobs JSON: {path}") from error
    if isinstance(payload, dict) and "jobs" in payload:
        payload = payload["jobs"]
    if isinstance(payload, dict) and "items" in payload:
        items = payload["items"]
        if not isinstance(items, list):
            raise MatchingError("Discovery report items must be a JSON list.")
        payload = [
            item.get("job")
            for item in items
            if isinstance(item, dict) and item.get("job")
        ]
    if not isinstance(payload, list):
        raise MatchingError("Jobs input must be a JSON list or discovery report.")
    try:
        return [JobPosting.model_validate(item) for item in payload]
    except (TypeError, ValidationError) as error:
        raise MatchingError(f"Jobs input contains invalid job data: {error}") from error


def create_live_models(settings: Settings) -> tuple[BaseChatModel, Embeddings]:
    """Create independently configured live LangChain providers."""
    if not settings.llm_provider or not settings.llm_model:
        raise MatchingError(
            "Set NEXTROLE_LLM_PROVIDER and NEXTROLE_LLM_MODEL for --live."
        )
    if not settings.embedding_provider or not settings.embedding_model:
        raise MatchingError(
            "Set NEXTROLE_EMBEDDING_PROVIDER and NEXTROLE_EMBEDDING_MODEL for --live."
        )
    local_llm = settings.llm_provider.casefold() == "ollama"
    local_embeddings = settings.embedding_provider.casefold() == "ollama"
    if not settings.llm_api_key and not local_llm:
        raise MatchingError("Set LLM_API_KEY for live matching.")
    chat = _create_chat(
        settings.llm_provider,
        settings.llm_model,
        settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
    )
    embedding_key = settings.embedding_api_key or settings.llm_api_key
    if not embedding_key and not local_embeddings:
        raise MatchingError("Set LLM_API_KEY or EMBEDDING_API_KEY for live matching.")
    embeddings = _create_embeddings(
        settings.embedding_provider,
        settings.embedding_model,
        embedding_key.get_secret_value() if embedding_key else None,
    )
    return chat, embeddings


def _create_chat(provider: str, model: str, api_key: str | None) -> BaseChatModel:
    normalized = provider.casefold()
    if normalized == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise MatchingError(
                "Install the Ollama LangChain integration with "
                "'python3 -m pip install langchain-ollama'."
            ) from error
        return ChatOllama(model=model, temperature=0)
    if normalized == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise MatchingError(
                "Install the openai LangChain extra before using the OpenAI provider."
            ) from error

        return ChatOpenAI(model=model, api_key=api_key, max_retries=0)
    if normalized == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as error:
            raise MatchingError(
                "Install the anthropic LangChain extra before using "
                "the Anthropic provider."
            ) from error

        return ChatAnthropic(model=model, api_key=api_key, max_retries=0)
    if normalized in {"google", "gemini"}:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as error:
            raise MatchingError(
                "Install the Google GenAI LangChain extra before using "
                "the Google provider."
            ) from error

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, max_retries=0
        )
    raise MatchingError(f"Unsupported LLM provider: {provider}")


def _create_embeddings(provider: str, model: str, api_key: str | None) -> Embeddings:
    normalized = provider.casefold()
    if normalized == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as error:
            raise MatchingError(
                "Install the Ollama LangChain integration with "
                "'python3 -m pip install langchain-ollama'."
            ) from error
        return OllamaEmbeddings(model=model)
    if normalized == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as error:
            raise MatchingError(
                "Install the openai LangChain extra before using OpenAI embeddings."
            ) from error

        return OpenAIEmbeddings(model=model, api_key=api_key, max_retries=0)
    if normalized in {"google", "gemini"}:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as error:
            raise MatchingError(
                "Install the Google GenAI LangChain extra before using "
                "Google embeddings."
            ) from error

        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    raise MatchingError(f"Unsupported embedding provider: {provider}")


def _deduplicate_jobs(jobs: Sequence[JobPosting]) -> tuple[list[JobPosting], int]:
    unique: list[JobPosting] = []
    seen: set[str] = set()
    duplicates = 0
    for job in jobs:
        key = str(job.url).casefold() if job.url else f"title:{job.title.casefold()}"
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(job)
    return unique, duplicates


def _job_query(job: JobPosting) -> str:
    data = job.model_dump(mode="json", exclude_none=True)
    return " ".join(str(value) for value in data.values())


def _has_job_evidence(job: JobPosting) -> bool:
    return bool(
        (job.description or "").strip()
        or any(item.strip() for item in job.requirements)
    )


def _prompt(profile: CandidateProfile, job: JobPosting, context: list[Document]) -> str:
    career_context = (
        "\n\n".join(
            f"[Career evidence {index}]\n{document.page_content}"
            for index, document in enumerate(context, start=1)
        )
        or "No career-history chunks were retrieved."
    )
    job_data = job.model_dump_json(exclude_none=True)
    return f"""Extract and assess the important job requirements against the candidate's
career evidence.

- Assess independently evaluable requirements separately. Preserve logical
  relationships, including alternatives, combined conditions, and dependencies.
- Evaluate requirements semantically rather than through keyword matching.
  Recognize equivalent capabilities and transferable experience without
  assuming they always fully satisfy a requirement.
- Assess each requirement against its stated scope. Do not introduce narrower
  qualifications or additional conditions unless explicitly required by the
  job evidence. If candidate evidence directly satisfies any explicitly
  accepted alternative, classify the requirement as met.
- For each assessment, identify supporting candidate evidence and assign:
  - met: Evidence establishes that the requirement is satisfied.
  - partially_met: Evidence establishes relevant but incomplete or transferable
    experience.
  - not_met: Evidence explicitly establishes that the requirement is not
    satisfied.
  - unknown: Available evidence is insufficient to determine whether the
    requirement is satisfied.
- Classify the evidence relationship as one of `direct`, `equivalent`,
  `transferable`, `absent`, or `contradictory`.
- Missing evidence must not be treated as evidence of a missing skill.
- Every met or partially_met assessment must reference specific candidate
  evidence. Do not invent or misattribute evidence.
- Evaluate mandatory eligibility constraints independently from technical fit.
- Before finalizing, check that requirement statuses agree with the cited
  evidence and that strengths, gaps, and uncertainties are consistent with the
  assessments.

Return only the required structured output.

<candidate_preferences>
{profile.preferences.model_dump_json()}
</candidate_preferences>

<candidate_career_evidence>
{career_context}
</candidate_career_evidence>

<job_evidence>
{job_data}
</job_evidence>"""


def _coerce_evaluation(response: Any, job: JobPosting) -> JobEvaluation:
    if isinstance(response, JobEvaluation):
        decision = response.model_copy(update={"job": job})
        return _with_deterministic_scores(decision)
    if isinstance(response, (EvaluationDecision, dict)):
        decision = EvaluationDecision.model_validate(response)
        return _with_deterministic_scores(
            JobEvaluation(
                job=job,
                **decision.model_dump(),
                demonstrated_fit=0,
                evidence_coverage=0,
                match_score=0,
            )
        )
    raise MatchingError("The model returned a non-structured evaluation.")


def _with_deterministic_scores(evaluation: JobEvaluation) -> JobEvaluation:
    assessments = [
        _normalize_assessment(assessment)
        for assessment in evaluation.requirement_assessments
    ]
    demonstrated_fit, evidence_coverage = _calculate_scores(assessments)
    return evaluation.model_copy(
        update={
            "requirement_assessments": assessments,
            "demonstrated_fit": demonstrated_fit,
            "evidence_coverage": evidence_coverage,
            # Keep the legacy field as a compatibility alias for demonstrated fit.
            "match_score": demonstrated_fit,
        }
    )


def _normalize_assessment(assessment: RequirementAssessment) -> RequirementAssessment:
    """Apply unambiguous evidence/status relationships in Python."""
    relationship = assessment.evidence_relationship
    status = assessment.status
    if relationship == "absent":
        status = "unknown"
    elif relationship == "contradictory":
        status = "not_met"
    elif relationship == "transferable" and status == "met":
        status = "partially_met"
    elif status in {"met", "partially_met"} and not assessment.candidate_evidence:
        status = "unknown"
    return assessment.model_copy(update={"status": status})


def _calculate_scores(
    assessments: list[RequirementAssessment],
) -> tuple[float, float]:
    """Return demonstrated fit and evidence coverage as percentages."""
    weights = {"required": 3, "preferred": 1, "unspecified": 1}
    status_values = {"met": 1.0, "partially_met": 0.5, "not_met": 0.0}
    total_weight = sum(weights[item.importance] for item in assessments)
    assessable = [item for item in assessments if item.status != "unknown"]
    assessable_weight = sum(weights[item.importance] for item in assessable)
    weighted_fit = sum(
        weights[item.importance] * status_values[item.status] for item in assessable
    )
    demonstrated_fit = (
        round(100 * weighted_fit / assessable_weight, 2) if assessable_weight else 0.0
    )
    evidence_coverage = (
        round(100 * assessable_weight / total_weight, 2) if total_weight else 0.0
    )
    return demonstrated_fit, evidence_coverage


def _extract_token_usage(response: Any) -> dict[str, int] | None:
    """Read provider usage metadata when a LangChain response exposes it."""
    metadata: Any = None
    if isinstance(response, dict):
        metadata = response.get("usage_metadata") or response.get("token_usage")
    else:
        metadata = getattr(response, "usage_metadata", None)
        if not metadata:
            response_metadata = getattr(response, "response_metadata", {})
            if isinstance(response_metadata, dict):
                metadata = response_metadata.get("token_usage")
    if not isinstance(metadata, dict):
        return None
    usage: dict[str, int] = {}
    for key in (
        "input_tokens",
        "prompt_tokens",
        "output_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        value = metadata.get(key)
        if isinstance(value, int):
            usage[key] = value
    return usage or None


def _merge_usage(current: dict[str, int] | None, new: dict[str, int]) -> dict[str, int]:
    merged = dict(current or {})
    for key, value in new.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _uncertain_evaluation(job: JobPosting, reason: str) -> JobEvaluation:
    return JobEvaluation(
        job=job,
        eligibility="uncertain",
        reason=reason,
        uncertainties=["The evaluation could not be validated."],
        demonstrated_fit=0,
        evidence_coverage=0,
        match_score=0,
    )


class FixtureChatModel:
    """Tiny LangChain-compatible fixture model for offline matching."""

    def with_structured_output(
        self, schema: type[EvaluationDecision]
    ) -> FixtureChatModel:
        return self

    def invoke(self, prompt: str) -> dict[str, Any]:
        del prompt
        return {
            "eligibility": "uncertain",
            "reason": "Fixture mode does not make a semantic eligibility claim.",
            "matching_strengths": [],
            "gaps": [],
            "uncertainties": ["Fixture response; run with --live for evaluation."],
        }


def _fallback_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    chunks: list[Document] = []
    step = chunk_size - chunk_overlap
    for index, start in enumerate(range(0, len(text), step)):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"source": "career_history", "chunk_index": index},
                )
            )
    return chunks


class _InMemoryVectorStore:
    """Offline fallback when optional Chroma packages are unavailable."""

    def __init__(self, embedding_function: Embeddings, **_: Any) -> None:
        self.embedding_function = embedding_function
        self.documents: list[Document] = []
        self.vectors: list[list[float]] = []

    def add_documents(self, documents: list[Document]) -> None:
        self.documents.extend(documents)
        self.vectors.extend(
            self.embedding_function.embed_documents(
                [document.page_content for document in documents]
            )
        )

    def similarity_search(self, query: str, k: int) -> list[Document]:
        vector = self.embedding_function.embed_query(query)
        ranked = sorted(
            zip(self.documents, self.vectors, strict=True),
            key=lambda pair: _cosine(vector, pair[1]),
            reverse=True,
        )
        return [document for document, _ in ranked[:k]]


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if not denominator:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator

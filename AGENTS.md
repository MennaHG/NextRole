# Repository conventions

- Use Python 3.11+ with type hints and small, single-purpose modules.
- Keep provider integrations isolated behind testable interfaces; never commit
  credentials or real candidate information.
- Run `pytest` and `ruff check .` before opening a change. Keep tests offline,
  deterministic, and independent of paid services.
- Update the README or relevant documentation when public behavior changes.
- Prefer clear names and straightforward validation over premature abstractions.

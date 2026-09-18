# Contributing to supermemoria

Thanks for your interest. Here's how to contribute effectively.

## What belongs here

New memory layers, new use cases, or backend integrations that extend the taxonomy without breaking its clarity.

**Good contributions:**
- A new use case in `cases/` (legal, finance, logistics, education, etc.)
- A vector backend for `SemanticMemory` or `CollectiveMemory`
- A persistence layer for `EpisodicMemory` (SQLite, Redis, etc.)
- Improvements to existing memory layers with clear rationale

**Not a good fit:**
- LLM wrappers (this is memory infrastructure, not an LLM framework)
- Heavy ML/GPU dependencies in core memory modules
- Behavior that belongs in `think()` or `act()`, not in memory

## Rules

1. Every new memory module must have tests in `tests/`
2. Every new use case must demonstrate at least 4 of the 8 memory layers
3. No external dependencies in `memory/` unless behind a backend interface
4. Keep modules under ~150 lines. If longer, split the responsibility.

## How to submit

1. Fork the repo
2. Create a branch: `feature/your-thing` or `fix/what-you-fixed`
3. Write tests first if adding to `memory/`
4. Submit a PR with a short description: what it is, why it belongs here

## Running tests

```bash
python tests/test_memory.py
```

All 25 tests must pass before merging.

# supermemoria

[![Tests](https://github.com/ulisesguras/supermemoria/actions/workflows/tests.yml/badge.svg)](https://github.com/ulisesguras/supermemoria/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

> An agent that remembers — not just what happened, but what it knows, how it feels, what it plans to do, and what the whole swarm has learned together.

Most AI agents have one kind of memory: a context window that resets.

**supermemoria** is a Python framework for building agents with nine distinct memory layers, organized across three cognitive dimensions — a synthesis of established cognitive-science memory models applied to AI agent architecture.

---

## The Nine Memory Layers

| Dimension | Layer | What it holds |
|-----------|-------|---------------|
| **By Duration** | `sensory` | Raw inputs, short TTL — what just arrived |
| **By Duration** | `working` | Active scratchpad, priority-evicted — what's being reasoned about now |
| **By Duration** | `episodic` | Long-term event log — what happened and when |
| **By Duration** | `semantic` | Long-term knowledge store — facts, domain knowledge, RAG-ready |
| **By Function** | `procedural` | Named skills and workflows — how to do things |
| **By Function** | `strategic` | Self-reflection, strategy tracking, meta-learning |
| **By Scope** | `emotional` | Affect signals: valence/arousal per subject — risk calibration |
| **By Scope** | `prospective` | Future intentions — time-based and event-triggered |
| **By Scope** | `collective` | Shared swarm knowledge — what the team knows together |

---

## Use Cases (6 domains, all 9 layers exercised)

| File | Domain | What it demonstrates |
|------|--------|----------------------|
| `cases/health_assistant.py` | Healthcare | Patient triage, affect-based tone calibration, medication reminders |
| `cases/customer_support.py` | Support | Frustration detection, escalation logic, collective bug reporting |
| `cases/energy_optimizer.py` | Energy / Grid | Sensor-driven dispatch, overload detection, grid learning |
| `cases/robotics_coordinator.py` | Robotics (Domain 5) | Fleet assignment, failure retry, strategy self-improvement |
| `cases/spatial_intelligence.py` | Spatial / Digital Twin (Domain 4) | IoT anomaly detection, emergency protocols, simulation |
| `cases/cognitive_agent.py` | Neuro-Inspired (Domain 1) | Adaptive reasoning, confidence calibration, human-in-the-loop |

---

## Architecture

```
supermemoria/
├── memory/
│   ├── sensory.py        ← raw input buffer, TTL-based decay
│   ├── working.py        ← priority scratchpad, capacity-limited
│   ├── episodic.py       ← append-only experience log
│   ├── semantic.py       ← fact store, pluggable vector backend
│   ├── procedural.py     ← named skill registry
│   ├── strategic.py      ← self-reflection, strategy registry, bias detection
│   ├── emotional.py      ← affect traces, valence/arousal per subject
│   ├── prospective.py    ← future intentions, time + event triggers
│   └── collective.py     ← shared swarm knowledge pool
│
├── agent/
│   └── __init__.py       ← BaseAgent: all 9 layers, ready to subclass
│
├── cases/
│   ├── health_assistant.py
│   ├── customer_support.py
│   ├── energy_optimizer.py
│   ├── robotics_coordinator.py
│   ├── spatial_intelligence.py
│   └── cognitive_agent.py
│
└── tests/
    └── test_memory.py    ← 32 tests, all passing
```

---

## Install

```bash
pip install -e .
```

Or with optional extras:

```bash
pip install -e ".[dev]"       # adds pytest + pytest-cov
pip install -e ".[vector]"    # adds chromadb
```

---

## Quickstart

```python
from supermemoria import BaseAgent, AgentConfig
from supermemoria.memory.collective import CollectiveMemory

agent = BaseAgent(
    config=AgentConfig(name="my-agent", role="assistant", domain="legal"),
    collective=CollectiveMemory(),
)

# Perceive raw input
agent.perceive("user_message", "contract has a problematic clause")

# Store domain knowledge
agent.learn("force_majeure_definition", "...", tags=["legal", "contracts"])

# Record what happened
agent.remember_episode(
    context="user flagged contract clause",
    action="applied deductive analysis",
    outcome="identified force majeure issue",
    success=True,
    tags=["legal", "contract_review"],
)

# Self-reflect on a decision
agent.reflect(
    decision="applied deductive reasoning to contract analysis",
    strategy="deductive",
    outcome="correct identification",
    lesson="deductive works well for contract clause analysis",
    performance_delta=0.05,
)

# Set a future intention
from time import time
agent.intend(
    description="follow up on contract review",
    trigger_type="event",
    trigger_value="client_response",
    action_description="send revised analysis",
)

# See full memory state
print(agent.memory_summary())
```

---

## What Strategic Memory Looks Like

```python
from supermemoria.memory.strategic import StrategicMemory

sm = StrategicMemory()
sm.register_strategy("deductive", "Rule-based reasoning", domain="legal")
sm.register_strategy("inductive", "Pattern-based reasoning", domain="legal")

# After many decisions, the agent knows which strategy works
sm._strategies["deductive"].update(success=True, delta=0.04)
sm._strategies["deductive"].update(success=True, delta=0.03)
sm._strategies["inductive"].update(success=False, delta=-0.05)

best = sm.best_strategy("legal", min_uses=1)
print(best.strategy_name)   # "deductive"
print(sm.systematic_biases())  # [("inductive", -0.05)]
```

---

## Plugging in a Vector Backend

```python
from supermemoria.memory.semantic import SemanticMemory
import chromadb

semantic = SemanticMemory()
semantic.use_vector_backend(
    embed_fn=your_embedding_function,
    db_client=chromadb.Client(),
)
```

No external dependencies required by default. Swap backends per module without touching agent logic.

---

## Run the Use Cases

```bash
python cases/health_assistant.py
python cases/customer_support.py
python cases/energy_optimizer.py
python cases/robotics_coordinator.py
python cases/spatial_intelligence.py
python cases/cognitive_agent.py
```

## Run the Tests

```bash
python tests/test_memory.py
```

32 tests across all 9 memory layers.

---

## Production Memory Failures This Framework Addresses

| Failure Mode | Which Layer Prevents It |
|---|---|
| Memory Poisoning | `collective.confidence` + consensus API |
| Memory Drift | `strategic.systematic_biases()` detection |
| Context Stuffing | `working` capacity + eviction by priority |
| Retrieval Hallucination | `semantic` pluggable backend with confidence scores |
| Memory Conflicts | `collective.consensus()` weighted agreement |
| Cache Staleness | `sensory` TTL decay + `working` priority updates |

---

## Design Principles

- **All 9 layers, always.** Every agent gets the complete taxonomy from day one.
- **Backend-agnostic.** Swap Redis, Chroma, Pinecone, or SQLite without touching agent logic.
- **Minimal dependencies.** The entire core runs on pure Python.
- **Real use cases first.** Every example solves a concrete problem from a real domain.
- **Gets better over time.** Strategic memory means agents that improve, not just agents that execute.

---

## Roadmap (contributions welcome)

- `memory/semantic_vector.py` — Chroma/Pinecone integration
- `memory/episodic_sql.py` — SQLite persistence
- `cases/advanced_materials.py` — research copilot (Domain 6)
- `cases/quantum_pipeline.py` — quantum circuit orchestration (Domain 8)
- `cases/space_mission.py` — autonomous mission agent (Domain 9)
- Multi-agent swarm simulation with full collective propagation

---

## License

Apache License 2.0

---

*Based on established cognitive science models of human memory architecture.*

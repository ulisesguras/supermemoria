"""
Strategic / Meta Memory
------------------------
The agent's memory of its own past decisions, reasoning patterns,
and performance over time. Enables self-reflection and improvement.

This is the layer that separates agents that repeat mistakes
from agents that get better.

Rooted in: Reflexion (arXiv:2303.11366)

Examples:
  - "My Tuesday demand forecasts are consistently 3% low"
  - "When user tone is urgent, I tend to over-escalate"
  - "Strategy X worked in 8/10 similar situations"

Enables: self-calibration, strategy selection, meta-learning.

Common Tools reference: LangGraph checkpointing, LangSmith,
LangChain callbacks, Weights & Biases, PostgreSQL feedback tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Reflection:
    reflection_id: str
    decision: str                          # What decision was made?
    strategy: str                          # What strategy was used?
    outcome: str                           # What happened?
    lesson: str                            # What should be remembered?
    domain: Optional[str] = None           # "forecasting", "triage", "routing", etc.
    performance_delta: Optional[float] = None  # +/- improvement vs baseline
    timestamp: float = field(default_factory=time)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "decision": self.decision,
            "strategy": self.strategy,
            "outcome": self.outcome,
            "lesson": self.lesson,
            "domain": self.domain,
            "performance_delta": self.performance_delta,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


@dataclass
class StrategyRecord:
    strategy_name: str
    description: str
    domain: str
    uses: int = 0
    successes: int = 0
    failures: int = 0
    avg_performance_delta: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses

    def update(self, success: bool, delta: float = 0.0) -> None:
        self.uses += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        # Running average of performance delta
        self.avg_performance_delta = (
            (self.avg_performance_delta * (self.uses - 1) + delta) / self.uses
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "description": self.description,
            "domain": self.domain,
            "uses": self.uses,
            "success_rate": round(self.success_rate, 3),
            "avg_performance_delta": round(self.avg_performance_delta, 4),
        }


class StrategicMemory:
    """
    Tracks agent decisions, outcomes, and lessons over time.

    Two subsystems:
      1. Reflections — append-only log of what was decided and learned
      2. Strategy registry — which strategies work best in which domains

    Use this to:
      - detect systematic biases in the agent's decisions
      - select the best strategy given a domain and history
      - surface lessons before making a new decision in a known domain
    """

    def __init__(self) -> None:
        self._reflections: List[Reflection] = []
        self._strategies: Dict[str, StrategyRecord] = {}
        self._id_counter = 0

    def reflect(
        self,
        decision: str,
        strategy: str,
        outcome: str,
        lesson: str,
        domain: Optional[str] = None,
        performance_delta: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Reflection:
        """Record a reflection after a decision is made and outcome observed."""
        self._id_counter += 1
        reflection = Reflection(
            reflection_id=f"ref_{self._id_counter:04d}",
            decision=decision,
            strategy=strategy,
            outcome=outcome,
            lesson=lesson,
            domain=domain,
            performance_delta=performance_delta,
            tags=tags or [],
        )
        self._reflections.append(reflection)

        # Auto-update strategy record if it exists
        if strategy in self._strategies and performance_delta is not None:
            success = (performance_delta >= 0)
            self._strategies[strategy].update(success, performance_delta)

        return reflection

    def register_strategy(
        self,
        name: str,
        description: str,
        domain: str,
    ) -> StrategyRecord:
        record = StrategyRecord(
            strategy_name=name,
            description=description,
            domain=domain,
        )
        self._strategies[name] = record
        return record

    def best_strategy(self, domain: str, min_uses: int = 3) -> Optional[StrategyRecord]:
        """
        Return the strategy with the highest success rate for a domain,
        among those used at least `min_uses` times.
        """
        candidates = [
            s for s in self._strategies.values()
            if s.domain == domain and s.uses >= min_uses
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: (s.success_rate, s.avg_performance_delta))

    def lessons_for_domain(self, domain: str) -> List[str]:
        """Surface all lessons learned in a given domain."""
        return [
            r.lesson for r in self._reflections
            if r.domain == domain
        ]

    def systematic_biases(self, threshold: float = -0.02) -> List[Tuple[str, float]]:
        """
        Detect strategies with a consistently negative performance delta.
        Returns list of (strategy_name, avg_delta) pairs.
        """
        return [
            (s.strategy_name, s.avg_performance_delta)
            for s in self._strategies.values()
            if s.uses >= 3 and s.avg_performance_delta <= threshold
        ]

    def recent_reflections(self, n: int = 10) -> List[Reflection]:
        return self._reflections[-n:]

    def by_domain(self, domain: str) -> List[Reflection]:
        return [r for r in self._reflections if r.domain == domain]

    def strategy_report(self) -> List[Dict[str, Any]]:
        return sorted(
            [s.to_dict() for s in self._strategies.values()],
            key=lambda x: -x["success_rate"],
        )

    def __len__(self) -> int:
        return len(self._reflections)

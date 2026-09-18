"""
BaseAgent
---------
A general-purpose agent with the complete memory taxonomy built in.
A synthesis of established cognitive-science memory models applied
to AI agent architecture.

9 memory layers across 3 cognitive dimensions:

  BY DURATION:
    sensory / working — short-term
    episodic / semantic — long-term

  BY FUNCTION:
    procedural — how to do things
    strategic — self-reflection, strategy tracking, meta-learning

  BY SCOPE:
    emotional / prospective — private to the agent
    collective — shared across the fleet

Override _setup(), think(), and act() to specialize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from supermemoria.memory.sensory import SensoryMemory
from supermemoria.memory.working import WorkingMemory
from supermemoria.memory.episodic import EpisodicMemory
from supermemoria.memory.semantic import SemanticMemory
from supermemoria.memory.procedural import ProceduralMemory
from supermemoria.memory.emotional import EmotionalMemory
from supermemoria.memory.prospective import ProspectiveMemory
from supermemoria.memory.collective import CollectiveMemory
from supermemoria.memory.strategic import StrategicMemory


@dataclass
class AgentConfig:
    name: str
    role: str
    description: str = ""
    domain: str = "general"
    sensory_capacity: int = 20
    sensory_ttl: float = 10.0
    working_capacity: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """
    Memory-rich agent with all 9 layers from the complete taxonomy.

      self.sensory      — raw inputs, TTL-based decay
      self.working      — active scratchpad, priority-evicted
      self.episodic     — append-only experience log
      self.semantic     — fact store, pluggable vector backend
      self.procedural   — named skill registry
      self.emotional    — affect traces (valence/arousal)
      self.prospective  — future intentions (time + event triggers)
      self.strategic    — self-reflection, strategy tracking, meta-learning
      self.collective   — shared swarm knowledge pool
    """

    def __init__(
        self,
        config: AgentConfig,
        collective: Optional[CollectiveMemory] = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self.role = config.role
        self.domain = config.domain

        self.sensory    = SensoryMemory(capacity=config.sensory_capacity, ttl_seconds=config.sensory_ttl)
        self.working    = WorkingMemory(capacity=config.working_capacity)
        self.episodic   = EpisodicMemory()
        self.semantic   = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.emotional  = EmotionalMemory()
        self.prospective = ProspectiveMemory()
        self.strategic  = StrategicMemory()
        self.collective = collective or CollectiveMemory()

        self._setup()

    def _setup(self) -> None:
        """Override: pre-load knowledge, register procedures/strategies."""
        pass

    def perceive(self, modality: str, content: Any) -> None:
        self.sensory.perceive(modality, content)

    def think(self, context: str) -> str:
        lessons = self.strategic.lessons_for_domain(self.domain)
        note = f" | Known lesson: {lessons[-1]}" if lessons else ""
        return f"[{self.name}] {context}{note}"

    def act(self, decision: str) -> str:
        outcome = f"executed: {decision}"
        self.episodic.record(
            context=self.working.get("current_task") or "unknown",
            action=decision,
            outcome=outcome,
            success=True,
        )
        return outcome

    def learn(self, key: str, value: Any, tags: Optional[List[str]] = None) -> None:
        _tags = tags or []
        self.semantic.store(key, value, source=self.name, tags=_tags)
        self.collective.contribute(contributor=self.name, key=key, value=value, tags=_tags)

    def remember_episode(
        self,
        context: str,
        action: str,
        outcome: str,
        success: Optional[bool] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.episodic.record(context=context, action=action, outcome=outcome,
                             success=success, tags=tags or [])

    def reflect(
        self,
        decision: str,
        strategy: str,
        outcome: str,
        lesson: str,
        performance_delta: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.strategic.reflect(
            decision=decision, strategy=strategy, outcome=outcome,
            lesson=lesson, domain=self.domain,
            performance_delta=performance_delta, tags=tags or [],
        )

    def intend(self, description: str, trigger_type: str, trigger_value: Any,
               action_description: str) -> None:
        self.prospective.intend(
            description=description, trigger_type=trigger_type,
            trigger_value=trigger_value, action_description=action_description,
        )

    def check_intentions(self, event: Optional[str] = None) -> List[Any]:
        return self.prospective.check(event=event)

    def memory_summary(self) -> Dict[str, Any]:
        biases = self.strategic.systematic_biases()
        return {
            "sensory_active":           len(self.sensory),
            "working_items":            len(self.working),
            "episodes":                 len(self.episodic),
            "facts":                    len(self.semantic),
            "procedures":               len(self.procedural),
            "reflections":              len(self.strategic),
            "affect_traces":            len(self.emotional),
            "pending_intentions":       len(self.prospective),
            "collective_records":       len(self.collective),
            "systematic_biases":        len(biases),
            "known_failures":           len(self.episodic.failures()),
        }

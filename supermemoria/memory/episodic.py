"""
Episodic Memory
---------------
A timeline of what the agent experienced and did.
Each episode is a self-contained record: context, action, outcome.

Enables: learning from past events, avoiding repeated mistakes,
building a sense of personal history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional
import json


@dataclass
class Episode:
    episode_id: str
    context: str                          # What situation was the agent in?
    action: str                           # What did it do?
    outcome: str                          # What happened?
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)
    success: Optional[bool] = None        # Was this episode a success?
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "success": self.success,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(**data)


class EpisodicMemory:
    """
    Append-only log of agent experiences.
    Supports filtering by tag, success, recency.
    """

    def __init__(self) -> None:
        self._episodes: List[Episode] = []
        self._id_counter = 0

    def record(
        self,
        context: str,
        action: str,
        outcome: str,
        success: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        self._id_counter += 1
        ep = Episode(
            episode_id=f"ep_{self._id_counter:05d}",
            context=context,
            action=action,
            outcome=outcome,
            success=success,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._episodes.append(ep)
        return ep

    def recent(self, n: int = 10) -> List[Episode]:
        return self._episodes[-n:]

    def by_tag(self, tag: str) -> List[Episode]:
        return [ep for ep in self._episodes if tag in ep.tags]

    def failures(self) -> List[Episode]:
        return [ep for ep in self._episodes if ep.success is False]

    def successes(self) -> List[Episode]:
        return [ep for ep in self._episodes if ep.success is True]

    def search(self, keyword: str) -> List[Episode]:
        kw = keyword.lower()
        return [
            ep for ep in self._episodes
            if kw in ep.context.lower()
            or kw in ep.action.lower()
            or kw in ep.outcome.lower()
        ]

    def export(self) -> List[Dict[str, Any]]:
        return [ep.to_dict() for ep in self._episodes]

    def __len__(self) -> int:
        return len(self._episodes)

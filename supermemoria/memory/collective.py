"""
Collective Memory
-----------------
Shared knowledge pool across multiple agents.
What the swarm knows together — not what any single agent remembers alone.

Enables: emergent learning, knowledge propagation, coordination without central control.

Designed to be backend-agnostic:
  - In-process dict (default, for single-process swarms)
  - Redis / shared DB for distributed agents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional


@dataclass
class SharedRecord:
    contributor: str             # Which agent wrote this?
    key: str
    value: Any
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contributor": self.contributor,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


class CollectiveMemory:
    """
    Append-write, multi-read shared memory for agent swarms.
    All agents write experiences; all agents can read the pool.

    For distributed agents: swap `_store` with a Redis or DB backend.
    """

    def __init__(self, backend: Optional[Any] = None) -> None:
        self._log: List[SharedRecord] = []
        self._index: Dict[str, List[SharedRecord]] = {}
        self._backend = backend

    def contribute(
        self,
        contributor: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> SharedRecord:
        record = SharedRecord(
            contributor=contributor,
            key=key,
            value=value,
            confidence=confidence,
            tags=tags or [],
        )
        self._log.append(record)
        self._index.setdefault(key, []).append(record)

        if self._backend:
            self._backend.write(record.to_dict())

        return record

    def recall(self, key: str, top_k: int = 1) -> List[SharedRecord]:
        """Return most recent records for a key."""
        records = self._index.get(key, [])
        return sorted(records, key=lambda r: -r.timestamp)[:top_k]

    def by_contributor(self, agent_id: str) -> List[SharedRecord]:
        return [r for r in self._log if r.contributor == agent_id]

    def by_tag(self, tag: str) -> List[SharedRecord]:
        return [r for r in self._log if tag in r.tags]

    def recent(self, n: int = 20) -> List[SharedRecord]:
        return self._log[-n:]

    def search(self, query: str) -> List[SharedRecord]:
        q = query.lower()
        return [
            r for r in self._log
            if q in r.key.lower()
            or (isinstance(r.value, str) and q in r.value.lower())
        ]

    def consensus(self, key: str) -> Optional[Any]:
        """
        Return the value whose group has the highest summed confidence score.

        Records are grouped by repr(value), and each group's weight is the SUM
        of its members' confidence scores. Agreement among several contributors
        therefore outweighs a single high-confidence outlier, making it harder
        for a malicious agent to poison the pool by injecting one very confident
        but incorrect value.
        """
        records = self._index.get(key, [])
        if not records:
            return None
        groups: Dict[str, list] = {}
        for r in records:
            groups.setdefault(repr(r.value), []).append(r)
        best_group = max(groups.values(), key=lambda g: sum(r.confidence for r in g))
        return best_group[0].value

    def __len__(self) -> int:
        return len(self._log)

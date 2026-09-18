"""
Semantic Memory
---------------
Long-term storage of facts, concepts, and domain knowledge.
Unlike episodic memory, semantic memory has no timestamps —
it stores what the agent *knows*, not what it *experienced*.

Enables: reasoning, retrieval, knowledge grounding.
Backends: in-memory dict (default), or swap for vector DB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Fact:
    key: str
    value: Any
    source: Optional[str] = None
    confidence: float = 1.0               # 0.0 - 1.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "tags": self.tags,
        }


class SemanticMemory:
    """
    A key-value store of structured knowledge.
    Supports tag-based retrieval and pluggable backends.

    For vector search: swap `self._store` with a Chroma/Pinecone client
    and override `search()` with embedding-based retrieval.
    """

    def __init__(self, backend: Optional[Any] = None) -> None:
        # Default: simple in-memory dict
        # Production: pass a vector DB client as backend
        self._store: Dict[str, Fact] = {}
        self._backend = backend  # Optional external backend hook

    def store(
        self,
        key: str,
        value: Any,
        source: Optional[str] = None,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> Fact:
        fact = Fact(key=key, value=value, source=source, confidence=confidence, tags=tags or [])
        self._store[key] = fact

        if self._backend:
            self._backend.upsert(key, fact.to_dict())

        return fact

    def recall(self, key: str) -> Optional[Any]:
        fact = self._store.get(key)
        return fact.value if fact else None

    def get_fact(self, key: str) -> Optional[Fact]:
        return self._store.get(key)

    def by_tag(self, tag: str) -> List[Fact]:
        return [f for f in self._store.values() if tag in f.tags]

    def search(self, query: str, top_k: int = 5) -> List[Fact]:
        """
        Simple keyword search by default.
        Override this method when using a vector backend.
        """
        q = query.lower()
        results = []
        for fact in self._store.values():
            score = 0
            if q in fact.key.lower():
                score += 2
            if isinstance(fact.value, str) and q in fact.value.lower():
                score += 1
            if any(q in tag.lower() for tag in fact.tags):
                score += 1
            if score > 0:
                results.append((score, fact))
        results.sort(key=lambda x: -x[0])
        return [f for _, f in results[:top_k]]

    def use_vector_backend(self, embed_fn: Callable, db_client: Any) -> None:
        """
        Plug in a vector backend.
        embed_fn: a function(text) -> List[float]
        db_client: Chroma, Pinecone, Weaviate, etc.
        """
        self._embed = embed_fn
        self._backend = db_client

    def all_facts(self) -> List[Fact]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

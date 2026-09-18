"""
Sensory Memory
--------------
Holds raw, unprocessed inputs for a very short window.
Like the human visual buffer — things you just perceived
but haven't decided to pay attention to yet.

Use: capture incoming signals before filtering.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Any, Deque, List


@dataclass
class SensoryTrace:
    modality: str        # "text", "tool_output", "user_message", "sensor", etc.
    content: Any
    received_at: float = field(default_factory=time)


class SensoryMemory:
    """
    A rolling buffer of recent raw inputs.
    Items decay quickly — nothing here survives long.
    """

    def __init__(self, capacity: int = 20, ttl_seconds: float = 10.0) -> None:
        self._buffer: Deque[SensoryTrace] = deque(maxlen=capacity)
        self.ttl = ttl_seconds

    def perceive(self, modality: str, content: Any) -> SensoryTrace:
        trace = SensoryTrace(modality=modality, content=content)
        self._buffer.append(trace)
        return trace

    def active(self) -> List[SensoryTrace]:
        """Return traces still within their TTL window."""
        now = time()
        return [t for t in self._buffer if (now - t.received_at) <= self.ttl]

    def flush(self) -> List[SensoryTrace]:
        """Drain all traces (e.g. before a reasoning cycle)."""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def __len__(self) -> int:
        return len(self.active())

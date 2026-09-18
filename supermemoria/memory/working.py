"""
Working Memory
--------------
The agent's active scratchpad during a task.
Holds current goal, active facts, and intermediate results.
Limited capacity — forces prioritization.

Analogous to human working memory: what you're currently thinking about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional


@dataclass
class WorkingItem:
    key: str
    value: Any
    priority: float = 1.0          # Higher = less likely to be evicted
    written_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def touch(self, new_value: Any) -> None:
        self.value = new_value
        self.updated_at = time()


class WorkingMemory:
    """
    A priority-aware scratchpad with a capacity limit.
    When full, lowest-priority items are evicted first.
    """

    def __init__(self, capacity: int = 10) -> None:
        self._items: Dict[str, WorkingItem] = {}
        self.capacity = capacity

    def set(self, key: str, value: Any, priority: float = 1.0) -> WorkingItem:
        if key in self._items:
            self._items[key].touch(value)
            self._items[key].priority = priority
            return self._items[key]

        if len(self._items) >= self.capacity:
            self._evict()

        item = WorkingItem(key=key, value=value, priority=priority)
        self._items[key] = item
        return item

    def get(self, key: str) -> Optional[Any]:
        item = self._items.get(key)
        return item.value if item else None

    def remove(self, key: str) -> bool:
        return bool(self._items.pop(key, None))

    def snapshot(self) -> Dict[str, Any]:
        return {k: v.value for k, v in self._items.items()}

    def _evict(self) -> None:
        if not self._items:
            return
        lowest = min(self._items.values(), key=lambda i: (i.priority, -i.updated_at))
        del self._items[lowest.key]

    def all_items(self) -> List[WorkingItem]:
        return list(self._items.values())

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

"""
Prospective Memory
------------------
Remembers things the agent *intends to do in the future*.
The missing layer in most agent frameworks.

Examples:
  - "remind user about invoice at end of day"
  - "re-check sensor reading in 10 minutes"
  - "follow up on open ticket after next user message"

Two modes:
  - time-based: trigger at a specific time
  - event-based: trigger when a condition is met
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Intention:
    intention_id: str
    description: str
    trigger_type: str                     # "time" | "event"
    trigger_value: Any                    # timestamp or event name/condition
    action_description: str
    handler: Optional[Callable] = None
    fired: bool = False
    created_at: float = field(default_factory=time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intention_id": self.intention_id,
            "description": self.description,
            "trigger_type": self.trigger_type,
            "trigger_value": self.trigger_value,
            "action_description": self.action_description,
            "fired": self.fired,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class ProspectiveMemory:
    """
    Stores future intentions and checks for matured triggers.
    Call `check(event=...)` each reasoning cycle to surface due intentions.
    """

    def __init__(self) -> None:
        self._intentions: List[Intention] = []
        self._id_counter = 0

    def intend(
        self,
        description: str,
        trigger_type: str,
        trigger_value: Any,
        action_description: str,
        handler: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Intention:
        self._id_counter += 1
        intention = Intention(
            intention_id=f"int_{self._id_counter:04d}",
            description=description,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            action_description=action_description,
            handler=handler,
            metadata=metadata or {},
        )
        self._intentions.append(intention)
        return intention

    def check(self, event: Optional[str] = None) -> List[Intention]:
        """
        Check which intentions are now due.
        - time-based: fires if current time >= trigger_value
        - event-based: fires if event matches trigger_value
        """
        now = time()
        due = []

        for intention in self._intentions:
            if intention.fired:
                continue

            if intention.trigger_type == "time":
                if now >= intention.trigger_value:
                    intention.fired = True
                    due.append(intention)
                    if intention.handler:
                        intention.handler(intention)

            elif intention.trigger_type == "event":
                if event and event == intention.trigger_value:
                    intention.fired = True
                    due.append(intention)
                    if intention.handler:
                        intention.handler(intention)

        return due

    def pending(self) -> List[Intention]:
        return [i for i in self._intentions if not i.fired]

    def fired(self) -> List[Intention]:
        return [i for i in self._intentions if i.fired]

    def __len__(self) -> int:
        return len(self.pending())

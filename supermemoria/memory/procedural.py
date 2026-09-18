"""
Procedural Memory
-----------------
Stores *how to do things* — skills, routines, playbooks.
Not facts (semantic) nor events (episodic) — but executable patterns.

Examples:
  - "how to escalate a customer complaint"
  - "how to run a demand forecast"
  - "how to handle a failed API call"

Enables: consistent behavior, skill composition, reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Procedure:
    name: str
    description: str
    steps: List[str]                        # Human-readable steps
    handler: Optional[Callable] = None      # Executable function, if available
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execute(self, *args, **kwargs) -> Any:
        if self.handler:
            return self.handler(*args, **kwargs)
        raise NotImplementedError(
            f"Procedure '{self.name}' has no executable handler. "
            "Steps: " + " | ".join(self.steps)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "tags": self.tags,
            "version": self.version,
            "metadata": self.metadata,
        }


class ProceduralMemory:
    """
    A registry of named procedures the agent can invoke.
    Think of it as the agent's skill library.
    """

    def __init__(self) -> None:
        self._procedures: Dict[str, Procedure] = {}

    def register(
        self,
        name: str,
        description: str,
        steps: List[str],
        handler: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Procedure:
        proc = Procedure(
            name=name,
            description=description,
            steps=steps,
            handler=handler,
            tags=tags or [],
            version=version,
            metadata=metadata or {},
        )
        self._procedures[name] = proc
        return proc

    def get(self, name: str) -> Optional[Procedure]:
        return self._procedures.get(name)

    def run(self, name: str, *args, **kwargs) -> Any:
        proc = self.get(name)
        if not proc:
            raise KeyError(f"No procedure named '{name}'")
        return proc.execute(*args, **kwargs)

    def by_tag(self, tag: str) -> List[Procedure]:
        return [p for p in self._procedures.values() if tag in p.tags]

    def list_all(self) -> List[str]:
        return list(self._procedures.keys())

    def search(self, query: str) -> List[Procedure]:
        q = query.lower()
        return [
            p for p in self._procedures.values()
            if q in p.name.lower() or q in p.description.lower()
        ]

    def __len__(self) -> int:
        return len(self._procedures)

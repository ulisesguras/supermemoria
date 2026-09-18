"""
Emotional Memory
----------------
Tracks affective states associated with events, people, and situations.
Influences confidence, urgency, and tone of agent responses.

Not about making agents "feel" things — about encoding
the *weight* of experiences: frustration, confidence, risk aversion.

Examples:
  - "interactions with user X tend to be high-friction"
  - "task type Y historically causes failures → caution warranted"
  - "this domain raises uncertainty → hedge responses"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional, Tuple


@dataclass
class AffectTrace:
    subject: str                   # What triggered this affect?
    valence: float                 # -1.0 (negative) to +1.0 (positive)
    arousal: float                 # 0.0 (calm) to 1.0 (intense)
    label: str                     # "frustration", "confidence", "curiosity", etc.
    context: Optional[str] = None
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "valence": self.valence,
            "arousal": self.arousal,
            "label": self.label,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class EmotionalMemory:
    """
    Stores affective traces and computes running sentiment
    toward subjects (users, tasks, domains).

    Use these signals to:
    - adjust confidence levels in responses
    - flag high-risk interactions
    - calibrate tone and caution
    """

    def __init__(self) -> None:
        self._traces: List[AffectTrace] = []

    def record(
        self,
        subject: str,
        valence: float,
        arousal: float = 0.5,
        label: str = "neutral",
        context: Optional[str] = None,
    ) -> AffectTrace:
        trace = AffectTrace(
            subject=subject,
            valence=max(-1.0, min(1.0, valence)),
            arousal=max(0.0, min(1.0, arousal)),
            label=label,
            context=context,
        )
        self._traces.append(trace)
        return trace

    def sentiment_toward(self, subject: str) -> Tuple[float, float]:
        """
        Returns (mean_valence, mean_arousal) for a given subject.
        Returns (0.0, 0.5) if unknown.
        """
        relevant = [t for t in self._traces if t.subject == subject]
        if not relevant:
            return (0.0, 0.5)
        avg_valence = sum(t.valence for t in relevant) / len(relevant)
        avg_arousal = sum(t.arousal for t in relevant) / len(relevant)
        return (avg_valence, avg_arousal)

    def high_friction_subjects(self, threshold: float = -0.3) -> List[str]:
        """Subjects with consistently negative valence."""
        subjects = set(t.subject for t in self._traces)
        return [s for s in subjects if self.sentiment_toward(s)[0] < threshold]

    def recent(self, n: int = 10) -> List[AffectTrace]:
        return self._traces[-n:]

    def by_label(self, label: str) -> List[AffectTrace]:
        return [t for t in self._traces if t.label == label]

    def __len__(self) -> int:
        return len(self._traces)

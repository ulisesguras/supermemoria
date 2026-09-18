"""
Use Case: Customer Support Agent
----------------------------------
Demonstrates memory-rich support: remembers users, learns from tickets,
anticipates follow-ups, and calibrates tone based on affect history.

Memory in action:
  sensory     — incoming messages
  working     — open ticket context
  episodic    — resolution history per user
  semantic    — product knowledge, known issues, policies
  procedural  — escalation, refund, troubleshooting flows
  emotional   — frustration signals per user
  prospective — follow-up promises ("I'll check back tomorrow")
  collective  — known bugs/outages shared across support agents
"""

from __future__ import annotations

from supermemoria import BaseAgent, AgentConfig
from supermemoria.memory.collective import CollectiveMemory


class SupportAgent(BaseAgent):

    def _setup(self) -> None:
        # Semantic: product knowledge
        self.semantic.store(
            "refund_policy",
            "Full refund within 30 days. Partial refund 30-90 days.",
            tags=["policy", "billing"],
        )
        self.semantic.store(
            "escalation_threshold",
            "Escalate if: 3+ failed attempts, billing dispute > $200, legal threat",
            tags=["policy", "escalation"],
        )

        # Procedural: support flows
        self.procedural.register(
            name="handle_complaint",
            description="Standard complaint resolution flow",
            steps=[
                "1. Acknowledge the user's frustration",
                "2. Query episodic memory — has this user had prior issues?",
                "3. Check semantic memory for relevant policies",
                "4. Propose resolution",
                "5. Record outcome in episodic memory",
                "6. Update emotional trace for this user",
            ],
            tags=["complaint"],
        )

        self.procedural.register(
            name="escalate_ticket",
            description="Hand off to human agent",
            steps=[
                "1. Flag ticket with urgency level",
                "2. Write full context summary to collective memory",
                "3. Record escalation in episodic memory",
                "4. Set prospective intention: follow up in 24h",
            ],
            tags=["escalation"],
        )

    def handle_ticket(self, user_id: str, message: str) -> dict:
        # Sensory: raw input
        self.perceive("user_message", {"user": user_id, "message": message})

        # Working: set ticket context
        self.working.set("active_user", user_id, priority=2.0)
        self.working.set("ticket_message", message, priority=2.0)

        # Emotional: check frustration level
        valence, arousal = self.emotional.sentiment_toward(user_id)
        is_frustrated = valence < -0.3

        # Episodic: check history
        past_tickets = self.episodic.search(user_id)
        repeat_user = len(past_tickets) > 2

        # Semantic: check escalation criteria
        escalation_policy = self.semantic.recall("escalation_threshold") or ""
        needs_escalation = (
            is_frustrated and repeat_user
            or "legal" in message.lower()
            or "billing dispute" in message.lower()
        )

        # Determine tone
        if is_frustrated:
            tone = "empathetic and careful"
        elif repeat_user:
            tone = "personalized — reference past history"
        else:
            tone = "friendly and efficient"

        # Record episode
        self.remember_episode(
            context=f"User {user_id}: {message}",
            action="ticket_triage",
            outcome="escalated" if needs_escalation else "in_progress",
            success=not needs_escalation,
            tags=["support", user_id],
        )

        # Collective: share known issues
        if "bug" in message.lower() or "broken" in message.lower():
            self.collective.contribute(
                contributor=self.name,
                key="known_issue",
                value=message[:120],
                tags=["bug_report"],
            )

        return {
            "user": user_id,
            "tone": tone,
            "escalate": needs_escalation,
            "repeat_user": repeat_user,
            "past_tickets": len(past_tickets),
            "frustration_level": round(abs(valence), 2) if is_frustrated else 0,
        }

    def record_user_affect(self, user_id: str, sentiment: str) -> None:
        mapping = {
            "frustrated": (-0.6, 0.8),
            "angry": (-0.9, 1.0),
            "neutral": (0.0, 0.3),
            "satisfied": (0.7, 0.3),
        }
        valence, arousal = mapping.get(sentiment, (0.0, 0.5))
        self.emotional.record(
            subject=user_id,
            valence=valence,
            arousal=arousal,
            label=sentiment,
        )


if __name__ == "__main__":
    collective = CollectiveMemory()
    agent = SupportAgent(
        config=AgentConfig(name="Support-1", role="customer_support"),
        collective=collective,
    )

    # Simulate history
    agent.record_user_affect("user_99", "frustrated")
    for _ in range(3):
        agent.episodic.record(
            context="user_99 prior complaint",
            action="attempted resolution",
            outcome="unresolved",
            success=False,
            tags=["user_99"],
        )

    result = agent.handle_ticket(
        user_id="user_99",
        message="This is the 4th time I'm contacting you. Billing dispute and the app is broken.",
    )

    print("Ticket Result:", result)
    print("Memory:", agent.memory_summary())

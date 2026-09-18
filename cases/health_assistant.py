"""
Use Case: Health & Wellness Assistant
--------------------------------------
Demonstrates how all 8 memory layers apply to a real health context.

Memory in action:
  sensory     — incoming symptoms, vitals, user messages
  working     — current health concern being evaluated
  episodic    — past health events and outcomes
  semantic    — medical knowledge, drug interactions, thresholds
  procedural  — how to escalate, how to generate a summary
  emotional   — frustration signals, patient anxiety patterns
  prospective — medication reminders, follow-up checks
  collective  — shared anonymized learnings across patients (opt-in)
"""

from __future__ import annotations

from supermemoria import BaseAgent, AgentConfig
from supermemoria.memory.collective import CollectiveMemory


class HealthAgent(BaseAgent):

    def _setup(self) -> None:
        # Semantic: baseline medical knowledge
        self.semantic.store("normal_heart_rate", "60-100 bpm", tags=["vitals"])
        self.semantic.store("normal_bp_systolic", "90-120 mmHg", tags=["vitals"])
        self.semantic.store("fever_threshold", "38.0°C / 100.4°F", tags=["vitals"])
        self.semantic.store(
            "escalation_criteria",
            "chest pain, difficulty breathing, loss of consciousness",
            tags=["safety", "escalation"],
        )

        # Procedural: how to handle a symptom report
        self.procedural.register(
            name="symptom_triage",
            description="Assess reported symptoms and decide urgency level",
            steps=[
                "1. Capture all reported symptoms into sensory buffer",
                "2. Check against known escalation criteria in semantic memory",
                "3. Query episodic memory for similar past episodes",
                "4. Assess emotional trace — is patient high-anxiety?",
                "5. Set working memory: urgency_level = low/medium/high",
                "6. If high urgency: trigger escalation procedure",
                "7. Record episode with outcome",
            ],
            tags=["triage", "safety"],
        )

        self.procedural.register(
            name="medication_reminder",
            description="Check and send medication reminders",
            steps=[
                "1. Query prospective memory for due reminders",
                "2. For each due intention: compose reminder message",
                "3. Record reminder sent in episodic memory",
                "4. Reset intention if recurring",
            ],
            tags=["reminders"],
        )

    def assess_symptoms(self, patient_id: str, symptoms: list[str]) -> dict:
        # Sensory: capture raw input
        for symptom in symptoms:
            self.perceive("symptom_report", {"patient": patient_id, "symptom": symptom})

        # Working: set active context
        self.working.set("current_patient", patient_id, priority=2.0)
        self.working.set("active_symptoms", symptoms, priority=2.0)

        # Check against escalation criteria
        escalation_criteria = self.semantic.recall("escalation_criteria") or ""
        high_risk = any(s.lower() in escalation_criteria for s in symptoms)

        urgency = "high" if high_risk else "medium" if len(symptoms) >= 3 else "low"
        self.working.set("urgency_level", urgency, priority=3.0)

        # Check past episodes for this patient
        past = self.episodic.search(patient_id)

        # Emotional: check if patient is high-friction / high-anxiety
        valence, arousal = self.emotional.sentiment_toward(patient_id)
        anxiety_flag = arousal > 0.7

        # Record episode
        self.remember_episode(
            context=f"Patient {patient_id} reported: {symptoms}",
            action="symptom_triage",
            outcome=f"Urgency assessed as {urgency}",
            success=True,
            tags=["triage", patient_id],
        )

        return {
            "patient": patient_id,
            "symptoms": symptoms,
            "urgency": urgency,
            "past_episodes": len(past),
            "anxiety_flag": anxiety_flag,
            "escalate": high_risk,
        }

    def schedule_medication_reminder(
        self, patient_id: str, medication: str, trigger_time: float
    ) -> None:
        self.intend(
            description=f"Remind {patient_id} to take {medication}",
            trigger_type="time",
            trigger_value=trigger_time,
            action_description=f"Send reminder: take {medication}",
        )

    def note_patient_affect(self, patient_id: str, label: str, valence: float) -> None:
        self.emotional.record(
            subject=patient_id,
            valence=valence,
            label=label,
            context="patient interaction",
        )


if __name__ == "__main__":
    from time import time

    collective = CollectiveMemory()
    agent = HealthAgent(
        config=AgentConfig(
            name="HealthBot-1",
            role="health_assistant",
            description="Triage and wellness support agent",
        ),
        collective=collective,
    )

    # Simulate patient interaction
    agent.note_patient_affect("patient_42", "anxiety", valence=-0.2)

    result = agent.assess_symptoms(
        patient_id="patient_42",
        symptoms=["headache", "fever", "chest pain"],
    )

    print("Assessment:", result)
    print()
    print("Memory Summary:", agent.memory_summary())

    # Schedule a medication reminder 5 seconds from now (demo)
    agent.schedule_medication_reminder("patient_42", "aspirin", time() + 5)
    print("Pending intentions:", len(agent.prospective.pending()))

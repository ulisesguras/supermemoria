"""
Use Case: Neuro-Inspired Cognitive Task Automation (Domain 1)
--------------------------------------------------------------
Adaptive agents that learn from outcomes and update behavior.
Human-in-the-loop agents simulating clinical or legal reasoning.
Context-aware prioritization in dynamic, noisy environments.

This is the most memory-intensive use case — all 9 layers are
fully exercised. The agent literally gets better over time.

Memory in action:
  sensory     — noisy input stream (documents, queries, signals)
  working     — current reasoning context, hypothesis in progress
  episodic    — full decision history with outcomes
  semantic    — domain knowledge, ground truth labels
  procedural  — reasoning workflows per task type
  emotional   — confidence calibration, uncertainty flags
  prospective — pending human review requests
  strategic   — which reasoning strategies are most accurate
  collective  — shared learnings across all cognitive agents
"""

from __future__ import annotations

import random

from supermemoria import BaseAgent, AgentConfig
from supermemoria.memory.collective import CollectiveMemory
from typing import Dict, List, Optional


class CognitiveAgent(BaseAgent):

    def _setup(self) -> None:
        # Semantic: domain knowledge
        self.semantic.store("confidence_threshold_for_autonomy", 0.85, tags=["policy", "calibration"])
        self.semantic.store("human_review_domains", ["legal", "medical", "financial"], tags=["policy"])
        self.semantic.store("reasoning_bias_tolerance", 0.03, tags=["policy", "bias"])

        # Procedural: reasoning flows
        self.procedural.register(
            name="adaptive_triage",
            description="Route task to correct reasoning strategy based on context",
            steps=[
                "1. Perceive task input into sensory memory",
                "2. Classify task domain via semantic memory",
                "3. Query strategic memory for best strategy in this domain",
                "4. Apply strategy with working memory as scratchpad",
                "5. Compute confidence score",
                "6. If confidence < threshold: request human review (prospective)",
                "7. Record full reasoning chain in episodic memory",
                "8. Reflect on outcome when feedback arrives",
            ],
            tags=["reasoning", "adaptive"],
        )

        # Strategic: register known reasoning strategies
        for strategy in ["deductive", "inductive", "analogical", "abductive"]:
            self.strategic.register_strategy(
                name=strategy,
                description=f"{strategy.capitalize()} reasoning approach",
                domain=self.domain,
            )

    def process_task(
        self,
        task_id: str,
        task_text: str,
        domain: str,
        true_label: Optional[str] = None,
    ) -> Dict:
        # Sensory: capture input
        self.perceive("task_input", {"id": task_id, "domain": domain, "text": task_text[:100]})
        self.working.set("current_task", task_id, priority=3.0)
        self.working.set("task_domain", domain, priority=2.0)

        # Strategic: pick best known strategy for this domain
        best = self.strategic.best_strategy(domain)
        strategy = best.strategy_name if best else "deductive"

        # Simulate confidence (in real agent: comes from LLM output)
        confidence = round(random.uniform(0.6, 1.0), 2)
        self.working.set("confidence", confidence, priority=2.5)

        # Emotional: low confidence = uncertainty flag
        if confidence < 0.75:
            self.emotional.record(
                subject=task_id,
                valence=-0.2,
                arousal=0.6,
                label="low_confidence",
                context=f"strategy={strategy}, conf={confidence}",
            )

        # Check if domain requires human review
        human_domains = self.semantic.recall("human_review_domains") or []
        threshold = self.semantic.recall("confidence_threshold_for_autonomy") or 0.85
        needs_review = domain in human_domains and confidence < threshold

        if needs_review:
            self.intend(
                description=f"Human review for {task_id}",
                trigger_type="event",
                trigger_value="human_available",
                action_description=f"Submit {task_id} in domain={domain} for human review",
            )

        # Record episode
        success = confidence >= threshold
        self.remember_episode(
            context=f"Task {task_id} in domain={domain}",
            action=f"applied {strategy} reasoning",
            outcome=f"confidence={confidence}, needs_review={needs_review}",
            success=success,
            tags=["cognitive", domain, task_id],
        )

        # If ground truth provided: reflect and update strategy
        if true_label is not None:
            correct = random.random() > 0.3  # simulate correctness
            delta = 0.05 if correct else -0.04
            self.reflect(
                decision=f"task {task_id}: applied {strategy}",
                strategy=strategy,
                outcome="correct" if correct else "incorrect",
                lesson=(
                    f"{strategy} works well for {domain}"
                    if correct else
                    f"{strategy} unreliable in {domain} — try inductive next"
                ),
                performance_delta=delta,
                tags=[domain, strategy],
            )
            # Update strategy record
            if strategy in self.strategic._strategies:
                self.strategic._strategies[strategy].update(correct, delta)

        # Collective: share confident outcomes
        if confidence >= threshold:
            self.collective.contribute(
                contributor=self.name,
                key=f"outcome:{domain}:{strategy}",
                value={"confidence": confidence, "task_type": task_text[:50]},
                confidence=confidence,
                tags=["cognitive", domain],
            )

        return {
            "task_id": task_id,
            "domain": domain,
            "strategy": strategy,
            "confidence": confidence,
            "needs_human_review": needs_review,
            "autonomous": not needs_review,
        }

    def improvement_report(self) -> Dict:
        """Show how the agent is learning over time."""
        biases = self.strategic.systematic_biases()
        return {
            "strategy_performance": self.strategic.strategy_report(),
            "systematic_biases": biases,
            "total_reflections": len(self.strategic),
            "failure_rate": round(
                len(self.episodic.failures()) / max(len(self.episodic), 1), 3
            ),
            "pending_human_reviews": len(self.prospective.pending()),
        }


if __name__ == "__main__":
    random.seed(42)
    collective = CollectiveMemory()
    agent = CognitiveAgent(
        config=AgentConfig(name="CogAgent-1", role="cognitive_automation", domain="legal"),
        collective=collective,
    )

    tasks = [
        ("t001", "Analyze contract clause for force majeure applicability", "legal", "applicable"),
        ("t002", "Review patient discharge summary for readmission risk", "medical", "high_risk"),
        ("t003", "Classify financial instrument for regulatory reporting", "financial", "tier_1"),
        ("t004", "Summarize deposition transcript for trial prep", "legal", None),
        ("t005", "Flag anomalous billing pattern in insurance claim", "financial", "anomalous"),
    ]

    for task_id, text, domain, label in tasks:
        result = agent.process_task(task_id, text, domain, label)
        print(f"{task_id}: {result}")

    print()
    print("Improvement Report:")
    for k, v in agent.improvement_report().items():
        print(f"  {k}: {v}")

    print()
    print("Memory:", agent.memory_summary())

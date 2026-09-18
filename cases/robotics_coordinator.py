"""
Use Case: Multi-Robot Coordination Agent (Domain 5 — Robotics)
---------------------------------------------------------------
Orchestrates fleets of warehouse, factory, or aerial robots.
Assigns tasks by location, health, and priority.
Adapts to sensor feedback and retries failed actions.

Memory in action:
  sensory     — real-time robot telemetry, sensor readings
  working     — current fleet state, active assignments
  episodic    — task history, failures, retries
  semantic    — robot specs, zone maps, task definitions
  procedural  — how to assign, retry, and rebalance fleet
  emotional   — risk signals (overheating, repeated failure)
  prospective — scheduled maintenance, recharge windows
  strategic   — which assignment strategies work in which zones
  collective  — shared fleet-level state across coordinator agents
"""

from __future__ import annotations

from supermemoria import BaseAgent, AgentConfig
from supermemoria.memory.collective import CollectiveMemory
from typing import Dict, List, Optional


ROBOT_STATES = ["idle", "moving", "working", "charging", "error"]


class RoboticsAgent(BaseAgent):

    def _setup(self) -> None:
        # Semantic: fleet knowledge
        self.semantic.store("max_battery_threshold", 20.0, tags=["fleet", "safety"])
        self.semantic.store("retry_limit", 3, tags=["fleet", "policy"])
        self.semantic.store(
            "priority_zones",
            {"A": "high", "B": "medium", "C": "low"},
            tags=["zones"],
        )

        # Procedural: coordination flows
        self.procedural.register(
            name="assign_task",
            description="Assign a task to the best available robot",
            steps=[
                "1. Query working memory for idle robots",
                "2. Filter by battery level > threshold",
                "3. Select robot closest to task zone (semantic)",
                "4. Check episodic for recent failures on this robot",
                "5. Assign and update working memory",
                "6. Record assignment in episodic memory",
            ],
            tags=["assignment"],
        )
        self.procedural.register(
            name="handle_failure",
            description="Retry or reassign after robot task failure",
            steps=[
                "1. Log failure in episodic memory with error tag",
                "2. Update emotional trace — flag robot as risk",
                "3. Check retry count in working memory",
                "4. If retries < limit: reassign to next available robot",
                "5. If retries >= limit: escalate to human operator",
                "6. Write pattern to collective memory",
            ],
            tags=["failure", "retry"],
        )

        # Strategic: register strategies for self-improvement
        self.strategic.register_strategy(
            name="nearest_robot_assignment",
            description="Assign task to robot physically closest to zone",
            domain="robotics",
        )
        self.strategic.register_strategy(
            name="healthiest_robot_assignment",
            description="Assign task to robot with highest battery + lowest error rate",
            domain="robotics",
        )

    def register_robot(self, robot_id: str, battery: float, zone: str) -> None:
        self.working.set(f"robot:{robot_id}", {
            "id": robot_id,
            "battery": battery,
            "zone": zone,
            "state": "idle",
            "retries": 0,
        }, priority=1.5)

    def assign_task(
        self,
        task_id: str,
        zone: str,
        priority: str = "medium",
        strategy: str = "nearest_robot_assignment",
    ) -> Dict:
        # Sensory: log incoming task
        self.perceive("task_request", {"task_id": task_id, "zone": zone, "priority": priority})
        self.working.set("active_task", task_id, priority=3.0)

        # Find idle robots from working memory
        all_items = self.working.all_items()
        robots = [
            item.value for item in all_items
            if isinstance(item.key, str) and item.key.startswith("robot:")
            and item.value.get("state") == "idle"
            and item.value.get("battery", 0) > (self.semantic.recall("max_battery_threshold") or 20)
        ]

        if not robots:
            self.remember_episode(
                context=f"Task {task_id} in zone {zone}",
                action="assign_task",
                outcome="no_robots_available",
                success=False,
                tags=["assignment", "no_capacity"],
            )
            return {"task_id": task_id, "status": "queued", "reason": "no_idle_robots"}

        # Pick best robot based on strategy
        best_robot = max(robots, key=lambda r: r.get("battery", 0))
        robot_id = best_robot["id"]

        # Update robot state
        self.working.set(f"robot:{robot_id}", {**best_robot, "state": "working"}, priority=2.0)

        # Record episode
        self.remember_episode(
            context=f"Task {task_id} in zone {zone}, priority={priority}",
            action=f"assigned to {robot_id} via {strategy}",
            outcome="assignment_success",
            success=True,
            tags=["assignment", zone, robot_id],
        )

        # Share to collective
        self.collective.contribute(
            contributor=self.name,
            key=f"assignment:{task_id}",
            value={"robot": robot_id, "zone": zone, "strategy": strategy},
            tags=["assignment"],
        )

        return {"task_id": task_id, "assigned_to": robot_id, "zone": zone, "strategy": strategy}

    def report_failure(self, robot_id: str, task_id: str, error: str) -> Dict:
        # Sensory: raw error signal
        self.perceive("robot_error", {"robot_id": robot_id, "error": error})

        # Emotional: flag this robot as risk
        self.emotional.record(
            subject=robot_id,
            valence=-0.7,
            arousal=0.8,
            label="task_failure",
            context=error,
        )

        # Episodic: record failure
        self.remember_episode(
            context=f"Robot {robot_id} on task {task_id}",
            action="task_execution",
            outcome=f"FAILURE: {error}",
            success=False,
            tags=["failure", robot_id, task_id],
        )

        # Update retry count
        robot_data = self.working.get(f"robot:{robot_id}") or {}
        retries = robot_data.get("retries", 0) + 1
        retry_limit = self.semantic.recall("retry_limit") or 3
        self.working.set(f"robot:{robot_id}", {**robot_data, "retries": retries, "state": "error"})

        # Strategic reflection
        self.reflect(
            decision=f"assigned {task_id} to {robot_id}",
            strategy="nearest_robot_assignment",
            outcome=f"failure after {retries} retries",
            lesson=f"Robot {robot_id} fails in high-load conditions — prefer healthiest_robot_assignment",
            performance_delta=-0.05,
            tags=["failure", robot_id],
        )

        if retries >= retry_limit:
            # Share escalation signal with collective
            self.collective.contribute(
                contributor=self.name,
                key=f"escalation:{robot_id}",
                value={"task": task_id, "retries": retries, "error": error},
                tags=["escalation", "failure"],
            )
            return {"status": "escalated", "robot_id": robot_id, "retries": retries}

        return {"status": "retry_queued", "robot_id": robot_id, "retries": retries}


if __name__ == "__main__":
    collective = CollectiveMemory()
    agent = RoboticsAgent(
        config=AgentConfig(name="FleetCoordinator-1", role="robotics_coordinator", domain="robotics"),
        collective=collective,
    )

    # Register fleet
    agent.register_robot("robot_01", battery=85.0, zone="A")
    agent.register_robot("robot_02", battery=45.0, zone="B")
    agent.register_robot("robot_03", battery=12.0, zone="A")  # below threshold

    # Assign tasks
    print(agent.assign_task("task_001", zone="A", priority="high"))
    print(agent.assign_task("task_002", zone="B", priority="medium"))

    # Simulate failure
    print(agent.report_failure("robot_01", "task_001", "motor_stall"))
    print(agent.report_failure("robot_01", "task_001", "motor_stall"))
    print(agent.report_failure("robot_01", "task_001", "motor_stall"))  # triggers escalation

    print()
    print("Memory Summary:", agent.memory_summary())
    print("Strategy Report:", agent.strategic.strategy_report())
    print("Lessons:", agent.strategic.lessons_for_domain("robotics"))

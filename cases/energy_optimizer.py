"""
Use Case: Energy Grid Optimizer
---------------------------------
Applies the memory taxonomy to energy demand forecasting
and load optimization in microgrids.

Memory in action:
  sensory     — real-time sensor readings, demand signals
  working     — current optimization problem state
  episodic    — past grid events, failures, interventions
  semantic    — grid topology, device specs, thresholds
  procedural  — load shedding, storage dispatch, demand response
  emotional   — risk signals (e.g. anomalous readings trigger caution)
  prospective — scheduled maintenance, peak demand windows
  collective  — shared learnings across grid agents
"""

from __future__ import annotations

from supermemoria import BaseAgent, AgentConfig
from supermemoria.memory.collective import CollectiveMemory


class EnergyAgent(BaseAgent):

    def _setup(self) -> None:
        self.semantic.store("peak_hours", "07:00-10:00, 17:00-21:00", tags=["grid", "timing"])
        self.semantic.store("battery_capacity_kwh", 100.0, tags=["storage"])
        self.semantic.store("max_load_kw", 500.0, tags=["grid", "limits"])
        self.semantic.store(
            "load_shedding_threshold",
            "Activate if load > 90% capacity for > 10 minutes",
            tags=["safety", "shedding"],
        )

        self.procedural.register(
            name="demand_response",
            description="Reduce load during peak periods",
            steps=[
                "1. Read current load from sensory buffer",
                "2. Compare against max_load_kw from semantic memory",
                "3. If load > 85%: query episodic for past interventions",
                "4. Select lowest-priority loads to reduce",
                "5. Record intervention in episodic memory",
                "6. Share pattern with collective memory",
            ],
            tags=["optimization", "demand_response"],
        )

        self.procedural.register(
            name="battery_dispatch",
            description="Dispatch battery storage to smooth peaks",
            steps=[
                "1. Check battery state-of-charge from working memory",
                "2. Assess forecasted demand in prospective memory",
                "3. Calculate optimal dispatch window",
                "4. Execute dispatch and record in episodic",
            ],
            tags=["storage", "dispatch"],
        )

    def process_reading(self, source: str, load_kw: float, battery_soc: float) -> dict:
        # Sensory: raw reading
        self.perceive("sensor", {"source": source, "load_kw": load_kw, "battery_soc": battery_soc})

        # Working: update state
        self.working.set("current_load_kw", load_kw, priority=3.0)
        self.working.set("battery_soc", battery_soc, priority=2.0)

        max_load = self.semantic.recall("max_load_kw") or 500.0
        load_ratio = load_kw / max_load

        # Emotional: flag anomalous readings as risk signals
        if load_ratio > 0.9:
            self.emotional.record(
                subject=source,
                valence=-0.7,
                arousal=0.9,
                label="overload_risk",
                context=f"Load at {load_ratio:.0%}",
            )

        # Check past overload events for this source
        past_overloads = self.episodic.by_tag("overload")

        # Decision
        if load_ratio > 0.9:
            action = "demand_response"
            success = False  # Not yet resolved
        elif load_ratio > 0.75 and battery_soc > 30:
            action = "battery_dispatch"
            success = True
        else:
            action = "monitor"
            success = True

        self.remember_episode(
            context=f"Grid {source}: load={load_kw}kW, soc={battery_soc}%",
            action=action,
            outcome=f"load_ratio={load_ratio:.2f}",
            success=success,
            tags=["grid", source] + (["overload"] if load_ratio > 0.9 else []),
        )

        # Share to collective
        self.collective.contribute(
            contributor=self.name,
            key=f"grid_reading:{source}",
            value={"load_kw": load_kw, "load_ratio": load_ratio, "action": action},
            tags=["grid"],
        )

        return {
            "source": source,
            "load_kw": load_kw,
            "load_ratio": round(load_ratio, 3),
            "action": action,
            "past_overloads": len(past_overloads),
            "battery_soc": battery_soc,
        }


if __name__ == "__main__":
    collective = CollectiveMemory()
    agent = EnergyAgent(
        config=AgentConfig(name="GridAgent-1", role="energy_optimizer"),
        collective=collective,
    )

    readings = [
        ("microgrid_A", 380.0, 65.0),
        ("microgrid_A", 450.0, 55.0),
        ("microgrid_A", 470.0, 40.0),
    ]

    for source, load, soc in readings:
        result = agent.process_reading(source, load, soc)
        print("Reading result:", result)

    print()
    print("Memory:", agent.memory_summary())
    print("Overload episodes:", len(agent.episodic.by_tag("overload")))

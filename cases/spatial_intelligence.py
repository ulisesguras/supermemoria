"""
Use Case: Spatial Intelligence / Digital Twin Coordination (Domain 4)
----------------------------------------------------------------------
Real-time monitoring of air quality, traffic, or infrastructure.
Simulates scenarios for urban design or emergency planning.
Controls smart environments via 3D spatial data.

Memory in action:
  sensory     — live IoT/sensor feeds, satellite data
  working     — current city state, active anomalies
  episodic    — incident history, intervention outcomes
  semantic    — city topology, threshold definitions, zone metadata
  procedural  — emergency protocols, evacuation routing
  emotional   — risk escalation flags per zone
  prospective — scheduled simulations, maintenance windows
  strategic   — which interventions worked in past incidents
  collective  — shared state across all city sector agents
"""

from __future__ import annotations

from supermemoria import BaseAgent, AgentConfig
from supermemoria.memory.collective import CollectiveMemory
from typing import Dict, List, Optional


class SpatialAgent(BaseAgent):

    def _setup(self) -> None:
        # Semantic: thresholds and zone knowledge
        self.semantic.store("aqi_hazardous", 300, tags=["air_quality", "threshold"])
        self.semantic.store("aqi_unhealthy", 150, tags=["air_quality", "threshold"])
        self.semantic.store("traffic_critical_density", 0.9, tags=["traffic", "threshold"])
        self.semantic.store(
            "evacuation_zones",
            {"zone_north": "route_A", "zone_south": "route_B", "zone_center": "route_C"},
            tags=["emergency", "routing"],
        )

        # Procedural: emergency protocols
        self.procedural.register(
            name="air_quality_alert",
            description="Respond to hazardous air quality readings",
            steps=[
                "1. Capture reading in sensory memory",
                "2. Compare against AQI thresholds in semantic",
                "3. Check episodic for prior alerts in this zone",
                "4. If level > hazardous: issue public advisory",
                "5. If level > critical: recommend evacuation",
                "6. Record intervention in episodic",
                "7. Broadcast warning to collective for all sector agents",
            ],
            tags=["air_quality", "emergency"],
        )
        self.procedural.register(
            name="traffic_rerouting",
            description="Reroute traffic during congestion or incidents",
            steps=[
                "1. Identify bottleneck zones from working memory",
                "2. Query semantic for alternate routes",
                "3. Check episodic for rerouting history",
                "4. Apply best-known strategy from strategic memory",
                "5. Simulate outcome before applying",
                "6. Record result",
            ],
            tags=["traffic"],
        )

        # Strategic: register intervention strategies
        self.strategic.register_strategy(
            name="preemptive_advisory",
            description="Issue warning before threshold is crossed",
            domain="spatial",
        )
        self.strategic.register_strategy(
            name="reactive_evacuation",
            description="Issue evacuation only after threshold exceeded",
            domain="spatial",
        )

    def process_sensor_reading(
        self,
        zone: str,
        sensor_type: str,
        value: float,
        unit: str,
    ) -> Dict:
        # Sensory: capture raw feed
        self.perceive("iot_sensor", {"zone": zone, "type": sensor_type, "value": value, "unit": unit})
        self.working.set(f"sensor:{zone}:{sensor_type}", value, priority=2.0)

        action = "monitor"
        alert_level = "none"
        strategy_used = "preemptive_advisory"

        if sensor_type == "aqi":
            hazardous = self.semantic.recall("aqi_hazardous") or 300
            unhealthy = self.semantic.recall("aqi_unhealthy") or 150

            if value >= hazardous:
                alert_level = "critical"
                action = "issue_evacuation_advisory"
                self.emotional.record(zone, valence=-0.9, arousal=1.0, label="critical_air_quality")
            elif value >= unhealthy:
                alert_level = "warning"
                action = "issue_public_warning"
                self.emotional.record(zone, valence=-0.5, arousal=0.7, label="unhealthy_air_quality")

        elif sensor_type == "traffic_density":
            critical = self.semantic.recall("traffic_critical_density") or 0.9
            if value >= critical:
                alert_level = "congestion"
                action = "traffic_reroute"
                self.emotional.record(zone, valence=-0.4, arousal=0.6, label="traffic_critical")

        # Episodic: record
        self.remember_episode(
            context=f"Zone {zone}: {sensor_type}={value}{unit}",
            action=action,
            outcome=f"alert_level={alert_level}",
            success=(alert_level == "none"),
            tags=["spatial", zone, sensor_type, alert_level],
        )

        # Strategic: reflect when an intervention is triggered
        if alert_level != "none":
            self.reflect(
                decision=f"{action} in {zone}",
                strategy=strategy_used,
                outcome=f"alert_level={alert_level}",
                lesson=f"Zone {zone} shows {sensor_type} spikes — consider preemptive threshold lowering",
                performance_delta=-0.03 if alert_level == "critical" else 0.01,
            )

        # Collective: broadcast anomalies
        if alert_level != "none":
            self.collective.contribute(
                contributor=self.name,
                key=f"alert:{zone}",
                value={"type": sensor_type, "value": value, "level": alert_level},
                confidence=0.95,
                tags=["alert", zone, sensor_type],
            )

        return {
            "zone": zone,
            "sensor_type": sensor_type,
            "value": value,
            "alert_level": alert_level,
            "action": action,
        }

    def simulate_scenario(self, scenario: str, zone: str) -> Dict:
        """Run a planning simulation (prospective + episodic)."""
        past = self.episodic.search(zone)
        lessons = self.strategic.lessons_for_domain("spatial")
        best_strategy = self.strategic.best_strategy("spatial")

        return {
            "scenario": scenario,
            "zone": zone,
            "past_incidents_in_zone": len(past),
            "recommended_strategy": best_strategy.strategy_name if best_strategy else "default",
            "lessons_available": len(lessons),
        }


if __name__ == "__main__":
    collective = CollectiveMemory()
    agent = SpatialAgent(
        config=AgentConfig(name="CityBrain-1", role="spatial_coordinator", domain="spatial"),
        collective=collective,
    )

    readings = [
        ("zone_north", "aqi", 145, "AQI"),
        ("zone_north", "aqi", 220, "AQI"),
        ("zone_north", "aqi", 320, "AQI"),   # critical
        ("zone_center", "traffic_density", 0.95, "ratio"),
    ]

    for zone, stype, val, unit in readings:
        result = agent.process_sensor_reading(zone, stype, val, unit)
        print(result)

    print()
    print("Simulation:", agent.simulate_scenario("emergency_evacuation", "zone_north"))
    print("Memory:", agent.memory_summary())

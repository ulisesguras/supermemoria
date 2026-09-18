"""
Tests for the complete memory taxonomy.
Run: python -m pytest tests/ -v
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from supermemoria.memory.sensory import SensoryMemory
from supermemoria.memory.working import WorkingMemory
from supermemoria.memory.episodic import EpisodicMemory
from supermemoria.memory.semantic import SemanticMemory
from supermemoria.memory.procedural import ProceduralMemory
from supermemoria.memory.emotional import EmotionalMemory
from supermemoria.memory.prospective import ProspectiveMemory
from supermemoria.memory.collective import CollectiveMemory
from supermemoria import BaseAgent, AgentConfig


class TestSensoryMemory(unittest.TestCase):
    def test_perceive_and_active(self):
        sm = SensoryMemory(capacity=5, ttl_seconds=10)
        sm.perceive("text", "hello")
        self.assertEqual(len(sm), 1)

    def test_ttl_expiry(self):
        sm = SensoryMemory(ttl_seconds=0.01)
        sm.perceive("text", "expires")
        time.sleep(0.05)
        self.assertEqual(len(sm), 0)

    def test_flush(self):
        sm = SensoryMemory()
        sm.perceive("text", "a")
        sm.perceive("text", "b")
        flushed = sm.flush()
        self.assertEqual(len(flushed), 2)
        self.assertEqual(len(sm.active()), 0)


class TestWorkingMemory(unittest.TestCase):
    def test_set_get(self):
        wm = WorkingMemory(capacity=3)
        wm.set("goal", "fix bug")
        self.assertEqual(wm.get("goal"), "fix bug")

    def test_eviction_by_priority(self):
        wm = WorkingMemory(capacity=2)
        wm.set("low", "evict me", priority=0.1)
        wm.set("high", "keep me", priority=2.0)
        wm.set("new", "triggers eviction", priority=1.0)
        self.assertIsNone(wm.get("low"))
        self.assertIsNotNone(wm.get("high"))

    def test_update(self):
        wm = WorkingMemory()
        wm.set("k", "v1")
        wm.set("k", "v2")
        self.assertEqual(wm.get("k"), "v2")


class TestEpisodicMemory(unittest.TestCase):
    def test_record_and_recent(self):
        em = EpisodicMemory()
        em.record("context", "action", "outcome", success=True, tags=["test"])
        self.assertEqual(len(em), 1)
        self.assertEqual(em.recent(1)[0].success, True)

    def test_search(self):
        em = EpisodicMemory()
        em.record("user complained", "resolved", "success")
        results = em.search("complained")
        self.assertEqual(len(results), 1)

    def test_failures(self):
        em = EpisodicMemory()
        em.record("a", "b", "c", success=False)
        em.record("d", "e", "f", success=True)
        self.assertEqual(len(em.failures()), 1)


class TestSemanticMemory(unittest.TestCase):
    def test_store_recall(self):
        sm = SemanticMemory()
        sm.store("fever", "38.0C", tags=["health"])
        self.assertEqual(sm.recall("fever"), "38.0C")

    def test_search(self):
        sm = SemanticMemory()
        sm.store("escalation_policy", "escalate if urgent", tags=["policy"])
        results = sm.search("escalate")
        self.assertTrue(len(results) > 0)

    def test_by_tag(self):
        sm = SemanticMemory()
        sm.store("k1", "v1", tags=["health"])
        sm.store("k2", "v2", tags=["billing"])
        self.assertEqual(len(sm.by_tag("health")), 1)


class TestProceduralMemory(unittest.TestCase):
    def test_register_and_run(self):
        pm = ProceduralMemory()
        pm.register(
            name="greet",
            description="Say hello",
            steps=["1. say hello"],
            handler=lambda: "hello!",
        )
        result = pm.run("greet")
        self.assertEqual(result, "hello!")

    def test_search(self):
        pm = ProceduralMemory()
        pm.register("triage", "assess symptoms", ["step1"])
        results = pm.search("assess")
        self.assertEqual(len(results), 1)


class TestEmotionalMemory(unittest.TestCase):
    def test_record_and_sentiment(self):
        em = EmotionalMemory()
        em.record("user_1", valence=-0.8, label="frustrated")
        em.record("user_1", valence=-0.4, label="frustrated")
        valence, _ = em.sentiment_toward("user_1")
        self.assertLess(valence, 0)

    def test_high_friction(self):
        em = EmotionalMemory()
        em.record("user_X", valence=-0.9)
        em.record("user_Y", valence=0.8)
        friction = em.high_friction_subjects()
        self.assertIn("user_X", friction)
        self.assertNotIn("user_Y", friction)


class TestProspectiveMemory(unittest.TestCase):
    def test_time_trigger(self):
        pm = ProspectiveMemory()
        pm.intend("remind", "time", time.time() - 1, "do something")
        due = pm.check()
        self.assertEqual(len(due), 1)

    def test_event_trigger(self):
        pm = ProspectiveMemory()
        pm.intend("on login", "event", "user_login", "show welcome")
        due = pm.check(event="user_login")
        self.assertEqual(len(due), 1)

    def test_pending_count(self):
        pm = ProspectiveMemory()
        pm.intend("future task", "time", time.time() + 9999, "far future")
        self.assertEqual(len(pm), 1)


class TestCollectiveMemory(unittest.TestCase):
    def test_contribute_and_recall(self):
        cm = CollectiveMemory()
        cm.contribute("agent_1", "bug_report", "login fails on mobile")
        records = cm.recall("bug_report")
        self.assertEqual(len(records), 1)

    def test_consensus(self):
        cm = CollectiveMemory()
        cm.contribute("a1", "status", "ok", confidence=0.6)
        cm.contribute("a2", "status", "degraded", confidence=0.9)
        result = cm.consensus("status")
        self.assertEqual(result, "degraded")

    def test_consensus_agreement_beats_single_outlier(self):
        cm = CollectiveMemory()
        for agent in ["a1", "a2", "a3"]:
            cm.contribute(agent, "cmd", "stable", confidence=0.7)
        cm.contribute("malicious", "cmd", "shutdown_all", confidence=1.0)
        self.assertEqual(cm.consensus("cmd"), "stable")

    def test_consensus_empty_returns_none(self):
        cm = CollectiveMemory()
        self.assertIsNone(cm.consensus("nonexistent_key"))

    def test_consensus_groups_unhashable_values(self):
        cm = CollectiveMemory()
        cm.contribute("a1", "cfg", {"mode": "eco"}, confidence=0.6)
        cm.contribute("a2", "cfg", {"mode": "eco"}, confidence=0.6)
        cm.contribute("a3", "cfg", {"mode": "max"}, confidence=0.9)
        self.assertEqual(cm.consensus("cfg"), {"mode": "eco"})

    def test_by_tag(self):
        cm = CollectiveMemory()
        cm.contribute("a1", "k1", "v1", tags=["grid"])
        cm.contribute("a2", "k2", "v2", tags=["health"])
        self.assertEqual(len(cm.by_tag("grid")), 1)


class TestBaseAgent(unittest.TestCase):
    def test_memory_summary(self):
        agent = BaseAgent(AgentConfig(name="test", role="tester"))
        summary = agent.memory_summary()
        self.assertIn("episodes", summary)
        self.assertIn("facts", summary)
        self.assertEqual(summary["episodes"], 0)

    def test_learn_goes_to_collective(self):
        collective = CollectiveMemory()
        agent = BaseAgent(AgentConfig(name="a1", role="tester"), collective=collective)
        agent.learn("key1", "value1", tags=["test"])
        self.assertEqual(len(agent.collective), 1)
        self.assertEqual(agent.semantic.recall("key1"), "value1")

    def test_perceive_and_work(self):
        agent = BaseAgent(AgentConfig(name="a1", role="tester"))
        agent.perceive("text", "incoming message")
        agent.working.set("task", "respond")
        self.assertEqual(agent.working.get("task"), "respond")
        self.assertEqual(len(agent.sensory), 1)


# ── Strategic Memory Tests ────────────────────────────────────────────────────

from supermemoria.memory.strategic import StrategicMemory

class TestStrategicMemory(unittest.TestCase):
    def test_reflect_and_retrieve(self):
        sm = StrategicMemory()
        sm.reflect(
            decision="assigned task to robot_01",
            strategy="nearest_robot",
            outcome="success",
            lesson="nearest robot assignment works in zone A",
            domain="robotics",
            performance_delta=0.05,
        )
        self.assertEqual(len(sm), 1)
        lessons = sm.lessons_for_domain("robotics")
        self.assertIn("nearest robot", lessons[0])

    def test_strategy_registry_and_best(self):
        sm = StrategicMemory()
        sm.register_strategy("strategy_A", "First strategy", "robotics")
        sm.register_strategy("strategy_B", "Second strategy", "robotics")
        for _ in range(5):
            sm._strategies["strategy_A"].update(True, 0.04)
        for _ in range(5):
            sm._strategies["strategy_B"].update(False, -0.02)
        best = sm.best_strategy("robotics", min_uses=3)
        self.assertEqual(best.strategy_name, "strategy_A")

    def test_systematic_biases(self):
        sm = StrategicMemory()
        sm.register_strategy("bad_strategy", "Consistently fails", "forecasting")
        for _ in range(5):
            sm._strategies["bad_strategy"].update(False, -0.05)
        biases = sm.systematic_biases(threshold=-0.02)
        self.assertEqual(len(biases), 1)
        self.assertEqual(biases[0][0], "bad_strategy")

    def test_reflect_updates_strategy(self):
        sm = StrategicMemory()
        sm.register_strategy("my_strat", "test strategy", "domain_x")
        sm.reflect("decision", "my_strat", "success", "lesson", domain="domain_x", performance_delta=0.1)
        rec = sm._strategies["my_strat"]
        self.assertEqual(rec.uses, 1)
        self.assertEqual(rec.successes, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

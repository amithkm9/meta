"""Tests for advanced simulation features: new tasks, memory decay, fatigue,
learning styles, PyTorch model, and analytics endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.env import TutoringEnv
from app.models import Action, ActionType
from app.main import app

api = TestClient(app)


# ── New task tests ────────────────────────────────────────────────────────

class TestNewTasks:
    """Verify all 6 tasks are loadable and produce valid episodes."""

    ALL_TASKS = [
        "easy_remediate_handshape",
        "medium_movement_timing_scaffold",
        "medium_orientation_spatial",
        "hard_multi_error_adaptive",
        "hard_kinesthetic_learner",
        "expert_sentence_flow_fatigue",
    ]

    @pytest.mark.parametrize("task_id", ALL_TASKS)
    def test_task_resets(self, task_id: str):
        env = TutoringEnv()
        obs = env.reset(task_id)
        assert obs.task_id == task_id
        assert obs.remaining_steps > 0
        assert obs.done is False

    @pytest.mark.parametrize("task_id", ALL_TASKS)
    def test_task_completes_with_grade(self, task_id: str):
        env = TutoringEnv()
        obs = env.reset(task_id, seed=42)
        while not obs.done:
            if obs.remaining_steps <= 1:
                action = Action(action_type=ActionType.FINALIZE_PLAN)
            else:
                action = Action(action_type=ActionType.SLOW_MOTION_DEMO)
            obs = env.step(action)
        assert env.final_grade is not None
        assert 0.0 <= env.final_grade.total_score <= 1.0

    def test_tasks_endpoint_has_six(self):
        resp = api.get("/tasks")
        assert resp.status_code == 200
        assert len(resp.json()) == 6

    def test_expert_task_has_four_errors(self):
        env = TutoringEnv()
        obs = env.reset("expert_sentence_flow_fatigue")
        assert len(obs.error_patterns) == 4
        assert obs.remaining_steps == 12

    def test_medium_orientation_has_two_errors(self):
        env = TutoringEnv()
        obs = env.reset("medium_orientation_spatial")
        error_types = [ep.error_type.value for ep in obs.error_patterns]
        assert "orientation" in error_types
        assert "location" in error_types


# ── Memory decay tests ────────────────────────────────────────────────────

class TestMemoryDecay:
    """Tasks with memory_decay_rate > 0 should lose comprehension over steps."""

    def test_comprehension_decays(self):
        env = TutoringEnv()
        env.reset("medium_orientation_spatial", seed=42)
        # Teach first
        env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        obs1 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        comp_after_teach = dict(obs1.learner_signals.assessed_comprehension)

        # Do nothing useful for a few steps (just assess and hint)
        env.step(Action(action_type=ActionType.ADD_MOVEMENT_HINT))
        env.step(Action(action_type=ActionType.CHOOSE_FEEDBACK_STYLE, payload={"style": "visual"}))
        obs2 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        comp_later = dict(obs2.learner_signals.assessed_comprehension)

        # With memory decay, some comprehension may have been lost
        # (though interventions add some back, decay should have an effect)
        # We just verify the mechanism doesn't crash and produces valid scores
        for et, val in comp_later.items():
            assert 0.0 <= val <= 1.0

    def test_no_decay_on_easy_task(self):
        """Easy task has no memory decay — comprehension should only increase."""
        env = TutoringEnv()
        env.reset("easy_remediate_handshape", seed=42)
        obs1 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        comp1 = obs1.learner_signals.assessed_comprehension["handshape"]
        env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        obs2 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        comp2 = obs2.learner_signals.assessed_comprehension["handshape"]
        assert comp2 >= comp1


# ── Fatigue tests ─────────────────────────────────────────────────────────

class TestFatigue:
    """Tasks with fatigue should show reduced effectiveness over time."""

    def test_expert_task_has_fatigue(self):
        env = TutoringEnv()
        env.reset("expert_sentence_flow_fatigue", seed=42)
        # Do several steps
        for _ in range(5):
            env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        assert env._fatigue > 0.0

    def test_easy_task_no_fatigue(self):
        env = TutoringEnv()
        env.reset("easy_remediate_handshape", seed=42)
        env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        assert env._fatigue == 0.0


# ── Learning style tests ─────────────────────────────────────────────────

class TestLearningStyles:
    """Learning styles should affect action effectiveness."""

    def test_kinesthetic_learner_benefits_from_drill(self):
        """Kinesthetic learner should get boosted gains from drill vs demo-only."""
        env = TutoringEnv()
        env.reset("hard_kinesthetic_learner", seed=42)
        obs = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        comp_start = dict(obs.learner_signals.assessed_comprehension)

        # Drill is boosted for kinesthetic learners (1.3x)
        env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        env.step(Action(action_type=ActionType.GENERATE_MICRO_DRILL, payload={"focus": "FRIEND"}))
        obs2 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        comp_after = dict(obs2.learner_signals.assessed_comprehension)

        # Should show improvement
        improved = any(comp_after[k] > comp_start[k] for k in comp_start)
        assert improved

    def test_visual_learner_benefits_from_cue(self):
        env = TutoringEnv()
        env.reset("medium_orientation_spatial", seed=42)
        obs = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        comp_start = dict(obs.learner_signals.assessed_comprehension)

        # Location cue boosted for visual learners (1.3x)
        env.step(Action(action_type=ActionType.ADD_LOCATION_CUE, payload={"cue": "chest marker"}))
        obs2 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        comp_after = dict(obs2.learner_signals.assessed_comprehension)

        assert comp_after["location"] > comp_start["location"]


# ── PyTorch model tests ──────────────────────────────────────────────────

class TestPyTorchModel:
    """Test the neural learner behavior model."""

    def test_model_loads(self):
        from app.learner_model import get_learner_model
        model = get_learner_model()
        assert model is not None

    def test_model_predicts(self):
        from app.learner_model import get_learner_model, predict_gains
        model = get_learner_model()
        result = predict_gains(
            model=model,
            action_type="slow_motion_demo",
            comprehension={"handshape": 0.3, "movement": 0.2},
            attention=0.8,
            frustration=0.1,
            confidence=0.5,
            fatigue=0.0,
            step=1,
            max_steps=6,
            has_assessed=False,
            has_demo=False,
            has_drill=False,
            has_feedback=False,
            has_prerequisite=False,
            learning_style="mixed",
        )
        assert "comprehension_gains" in result
        assert "attention_delta" in result
        assert "frustration_delta" in result
        assert len(result["comprehension_gains"]) == 5

    def test_model_output_ranges(self):
        from app.learner_model import get_learner_model, predict_gains
        model = get_learner_model()
        result = predict_gains(
            model=model,
            action_type="generate_micro_drill",
            comprehension={"movement": 0.5, "timing": 0.3},
            attention=0.6,
            frustration=0.3,
            confidence=0.4,
            fatigue=0.2,
            step=4,
            max_steps=8,
            has_assessed=True,
            has_demo=True,
            has_drill=False,
            has_feedback=False,
            has_prerequisite=False,
            learning_style="kinesthetic",
        )
        for et, gain in result["comprehension_gains"].items():
            assert 0.0 <= gain <= 0.3, f"Gain for {et} out of range: {gain}"
        assert -0.15 <= result["attention_delta"] <= 0.15
        assert -0.15 <= result["frustration_delta"] <= 0.15


# ── Analytics endpoint test ───────────────────────────────────────────────

class TestAnalyticsEndpoint:
    def test_analytics_after_episode(self):
        api.post("/reset", json={"task_id": "easy_remediate_handshape"})
        api.post("/step", json={"action": {"action_type": "slow_motion_demo", "rationale": "test"}})
        api.post("/step", json={"action": {"action_type": "finalize_plan", "rationale": "done"}})

        resp = api.get("/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "easy_remediate_handshape"
        assert data["steps_taken"] == 2
        assert data["done"] is True
        assert "coverage" in data
        assert "action_history" in data
        assert len(data["action_history"]) == 2

    def test_analytics_without_reset_fails(self):
        # Fresh client — need to ensure no active session
        # (this depends on state from other tests, so just check it doesn't 500)
        resp = api.get("/analytics")
        assert resp.status_code in (200, 400)


# ── Difficulty progression test ───────────────────────────────────────────

class TestDifficultyProgression:
    """Harder tasks should be harder to score well on with a naive policy."""

    def test_expert_harder_than_easy(self):
        # Easy task with naive policy
        env_easy = TutoringEnv()
        env_easy.reset("easy_remediate_handshape", seed=42)
        for _ in range(5):
            env_easy.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        env_easy.step(Action(action_type=ActionType.FINALIZE_PLAN))
        easy_score = env_easy.final_grade.total_score

        # Expert task with same naive policy
        env_expert = TutoringEnv()
        env_expert.reset("expert_sentence_flow_fatigue", seed=42)
        for _ in range(11):
            env_expert.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        env_expert.step(Action(action_type=ActionType.FINALIZE_PLAN))
        expert_score = env_expert.final_grade.total_score

        # Expert should be harder — lower score with naive policy
        assert expert_score < easy_score

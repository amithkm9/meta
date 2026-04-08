"""Tests for the core TutoringEnv with dynamic learner simulation."""

import pytest

from app.env import TutoringEnv
from app.models import Action, ActionType


@pytest.fixture
def env():
    return TutoringEnv()


def test_reset_returns_valid_observation(env: TutoringEnv):
    obs = env.reset("easy_remediate_handshape")
    assert obs.task_id == "easy_remediate_handshape"
    assert obs.remaining_steps == 6
    assert obs.step_count == 0
    assert obs.done is False
    assert obs.learner_signals.engagement != ""


def test_step_advances_state(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    obs = env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO, rationale="demo"))
    assert obs.step_count == 1
    assert obs.remaining_steps == 5
    assert obs.done is False


def test_assessment_reveals_comprehension(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    obs = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT, rationale="check"))
    assert obs.learner_signals.assessed_comprehension is not None
    assert "handshape" in obs.learner_signals.assessed_comprehension


def test_comprehension_hidden_without_assessment(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    obs = env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO, rationale="demo"))
    assert obs.learner_signals.assessed_comprehension is None


def test_intervention_improves_comprehension(env: TutoringEnv):
    env.reset("medium_movement_timing_scaffold", seed=42)
    # Assess before
    obs1 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
    comp_before = obs1.learner_signals.assessed_comprehension
    assert comp_before is not None

    # Teach
    env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
    env.step(Action(action_type=ActionType.ADD_MOVEMENT_HINT))

    # Assess after
    obs2 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
    comp_after = obs2.learner_signals.assessed_comprehension
    assert comp_after is not None
    assert comp_after["movement"] > comp_before["movement"]


def test_finalize_terminates(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
    obs = env.step(Action(action_type=ActionType.FINALIZE_PLAN, rationale="done"))
    assert obs.done is True
    assert env.final_grade is not None


def test_max_steps_terminates(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    for i in range(6):
        obs = env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO, rationale=f"step {i}"))
    assert obs.done is True


def test_reward_in_range(env: TutoringEnv):
    env.reset("medium_movement_timing_scaffold")
    actions = [
        ActionType.QUICK_ASSESSMENT,
        ActionType.SLOW_MOTION_DEMO,
        ActionType.ADD_MOVEMENT_HINT,
        ActionType.GENERATE_MICRO_DRILL,
        ActionType.CHOOSE_FEEDBACK_STYLE,
        ActionType.QUICK_ASSESSMENT,
        ActionType.FINALIZE_PLAN,
    ]
    for at in actions:
        obs = env.step(Action(action_type=at, rationale="test"))
        r = env.last_step_reward
        assert r is not None
        assert 0.0 <= r <= 1.0, f"Reward {r} out of range for {at}"
        if obs.done:
            break


def test_final_score_in_range(env: TutoringEnv):
    env.reset("hard_multi_error_adaptive")
    actions = [
        ActionType.QUICK_ASSESSMENT,
        ActionType.SELECT_PREREQUISITE_SIGN,
        ActionType.SLOW_MOTION_DEMO,
        ActionType.ADD_LOCATION_CUE,
        ActionType.ADD_MOVEMENT_HINT,
        ActionType.GENERATE_MICRO_DRILL,
        ActionType.CHOOSE_FEEDBACK_STYLE,
        ActionType.QUICK_ASSESSMENT,
        ActionType.REVISION_LOOP,
        ActionType.FINALIZE_PLAN,
    ]
    for at in actions:
        obs = env.step(Action(action_type=at, rationale="test"))
        if obs.done:
            break
    assert obs.done
    grade = env.final_grade
    assert grade is not None
    assert 0.0 <= grade.total_score <= 1.0


def test_duplicate_actions_reduce_reward(env: TutoringEnv):
    """Repeating the same action should yield diminishing rewards."""
    env.reset("easy_remediate_handshape")
    env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
    r1 = env.last_step_reward
    env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
    r2 = env.last_step_reward
    env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
    r3 = env.last_step_reward
    # Third repetition should have lower reward due to efficiency penalty
    assert r3 < r1


def test_step_after_done_raises(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    env.step(Action(action_type=ActionType.FINALIZE_PLAN))
    with pytest.raises(RuntimeError, match="done"):
        env.step(Action(action_type=ActionType.FINALIZE_PLAN))


def test_state_property(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    s = env.state
    assert s.episode_id is not None
    assert s.task_id == "easy_remediate_handshape"
    assert s.step_count == 0


def test_seeded_reproducibility(env: TutoringEnv):
    """Same seed should produce same results."""
    env.reset("medium_movement_timing_scaffold", seed=123)
    env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
    obs1 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))

    env.reset("medium_movement_timing_scaffold", seed=123)
    env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
    obs2 = env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))

    assert obs1.learner_signals.assessed_comprehension == obs2.learner_signals.assessed_comprehension


def test_grade_shows_comprehension_change(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
    env.step(Action(action_type=ActionType.GENERATE_MICRO_DRILL))
    env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
    env.step(Action(action_type=ActionType.FINALIZE_PLAN))
    grade = env.final_grade
    assert grade is not None
    assert grade.comprehension_before != grade.comprehension_after
    assert grade.comprehension_after["handshape"] > grade.comprehension_before["handshape"]

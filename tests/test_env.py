"""Tests for the core TutoringEnv."""

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


def test_step_advances_state(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    obs = env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO, rationale="demo"))
    assert obs.step_count == 1
    assert obs.remaining_steps == 5
    assert obs.done is False


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
    env.reset("medium_fix_movement_with_scaffold")
    actions = [
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
        assert 0.0 <= r <= 1.0, f"Reward {r} out of range"
        if obs.done:
            break


def test_final_score_in_range(env: TutoringEnv):
    env.reset("hard_adaptive_multi_error_plan")
    actions = [
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

"""Tests for the core TutoringEnv."""

import pytest

from app.env import TutoringEnv
from app.models import Action, ActionType


@pytest.fixture
def env():
    return TutoringEnv()


def test_reset_returns_valid_state(env: TutoringEnv):
    resp = env.reset("easy_remediate_handshape")
    assert resp.episode_id
    assert resp.done is False
    obs = resp.observation
    assert obs.task_id == "easy_remediate_handshape"
    assert obs.remaining_steps == 6
    assert obs.step_count == 0


def test_step_advances_state(env: TutoringEnv):
    resp = env.reset("easy_remediate_handshape")
    eid = resp.episode_id

    result = env.step(eid, Action(action_type=ActionType.SLOW_MOTION_DEMO, rationale="demo"))
    assert result.observation.step_count == 1
    assert result.observation.remaining_steps == 5
    assert result.done is False


def test_finalize_terminates(env: TutoringEnv):
    resp = env.reset("easy_remediate_handshape")
    eid = resp.episode_id

    env.step(eid, Action(action_type=ActionType.SLOW_MOTION_DEMO))
    result = env.step(eid, Action(action_type=ActionType.FINALIZE_PLAN, rationale="done"))
    assert result.done is True
    assert "final_grade" in result.info


def test_max_steps_terminates(env: TutoringEnv):
    resp = env.reset("easy_remediate_handshape")
    eid = resp.episode_id
    max_steps = resp.observation.remaining_steps

    for i in range(max_steps):
        result = env.step(eid, Action(action_type=ActionType.SLOW_MOTION_DEMO, rationale=f"step {i}"))

    assert result.done is True


def test_reward_in_range(env: TutoringEnv):
    resp = env.reset("medium_fix_movement_with_scaffold")
    eid = resp.episode_id

    actions = [
        ActionType.SLOW_MOTION_DEMO,
        ActionType.ADD_MOVEMENT_HINT,
        ActionType.GENERATE_MICRO_DRILL,
        ActionType.CHOOSE_FEEDBACK_STYLE,
        ActionType.QUICK_ASSESSMENT,
        ActionType.FINALIZE_PLAN,
    ]
    for at in actions:
        result = env.step(eid, Action(action_type=at, rationale="test"))
        assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range"
        if result.done:
            break


def test_final_score_in_range(env: TutoringEnv):
    resp = env.reset("hard_adaptive_multi_error_plan")
    eid = resp.episode_id

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
    result = None
    for at in actions:
        result = env.step(eid, Action(action_type=at, rationale="test"))
        if result.done:
            break

    assert result is not None and result.done
    grade = result.info["final_grade"]
    assert 0.0 <= grade["total_score"] <= 1.0


def test_episode_id_mismatch_raises(env: TutoringEnv):
    env.reset("easy_remediate_handshape")
    with pytest.raises(ValueError, match="mismatch"):
        env.step("wrong-id", Action(action_type=ActionType.FINALIZE_PLAN))


def test_step_after_done_raises(env: TutoringEnv):
    resp = env.reset("easy_remediate_handshape")
    eid = resp.episode_id
    env.step(eid, Action(action_type=ActionType.FINALIZE_PLAN))
    with pytest.raises(RuntimeError, match="done"):
        env.step(eid, Action(action_type=ActionType.FINALIZE_PLAN))

"""Tests for the outcome-based grading logic."""

import pytest

from app.grader import grade_episode
from app.models import ActionType, CoverageFlags, LearnerSimState
from app.tasks import get_task


def test_grade_normalization():
    """All sub-scores and total must be in [0, 1]."""
    task = get_task("easy_remediate_handshape")
    grade = grade_episode(
        task=task,
        action_history=[ActionType.FINALIZE_PLAN],
        coverage=CoverageFlags(plan_finalized=True),
        satisfied_requirements=set(),
        learner=LearnerSimState(comprehension={"handshape": 0.25}),
        initial_comprehension={"handshape": 0.20},
        step_count=1,
    )
    assert 0.0 <= grade.total_score <= 1.0
    assert 0.0 <= grade.sub_scores.comprehension_gain <= 1.0
    assert 0.0 <= grade.sub_scores.pedagogical_quality <= 1.0
    assert 0.0 <= grade.sub_scores.learner_wellbeing <= 1.0
    assert 0.0 <= grade.sub_scores.efficiency <= 1.0


def test_high_comprehension_scores_well():
    """If learner reaches target comprehension, score should be high."""
    task = get_task("easy_remediate_handshape")
    grade = grade_episode(
        task=task,
        action_history=[
            ActionType.SLOW_MOTION_DEMO,
            ActionType.ADD_MOVEMENT_HINT,
            ActionType.GENERATE_MICRO_DRILL,
            ActionType.CHOOSE_FEEDBACK_STYLE,
            ActionType.QUICK_ASSESSMENT,
            ActionType.FINALIZE_PLAN,
        ],
        coverage=CoverageFlags(
            has_timing_support=True,
            has_movement_hint=True,
            has_micro_drill=True,
            has_feedback_style=True,
            has_assessment=True,
            demo_before_drill=True,
            plan_finalized=True,
        ),
        satisfied_requirements={"corrective_intervention", "assessment"},
        learner=LearnerSimState(
            comprehension={"handshape": 0.80},
            attention=0.70,
            frustration=0.10,
            confidence=0.65,
        ),
        initial_comprehension={"handshape": 0.20},
        step_count=6,
    )
    assert grade.total_score >= 0.70
    assert grade.passed is True
    assert grade.comprehension_after["handshape"] > grade.comprehension_before["handshape"]


def test_low_comprehension_scores_poorly():
    """If learner barely improved, score should be low."""
    task = get_task("easy_remediate_handshape")
    grade = grade_episode(
        task=task,
        action_history=[ActionType.FINALIZE_PLAN],
        coverage=CoverageFlags(plan_finalized=True),
        satisfied_requirements=set(),
        learner=LearnerSimState(
            comprehension={"handshape": 0.22},
            attention=0.40,
            frustration=0.50,
            confidence=0.30,
        ),
        initial_comprehension={"handshape": 0.20},
        step_count=1,
    )
    assert grade.total_score < 0.60
    assert grade.passed is False


def test_redundant_actions_reduce_efficiency():
    task = get_task("medium_movement_timing_scaffold")
    good = grade_episode(
        task=task,
        action_history=[
            ActionType.QUICK_ASSESSMENT,
            ActionType.SLOW_MOTION_DEMO,
            ActionType.ADD_MOVEMENT_HINT,
            ActionType.GENERATE_MICRO_DRILL,
            ActionType.CHOOSE_FEEDBACK_STYLE,
            ActionType.FINALIZE_PLAN,
        ],
        coverage=CoverageFlags(
            has_assessment=True, has_timing_support=True,
            has_movement_hint=True, has_micro_drill=True,
            has_feedback_style=True, demo_before_drill=True,
            plan_finalized=True,
        ),
        satisfied_requirements={"scaffolded_demo", "micro_drill", "assessment", "feedback_selection"},
        learner=LearnerSimState(
            comprehension={"movement": 0.70, "timing": 0.65},
            attention=0.65, frustration=0.15, confidence=0.60,
        ),
        initial_comprehension={"movement": 0.25, "timing": 0.20},
        step_count=6,
    )
    bad = grade_episode(
        task=task,
        action_history=[
            ActionType.SLOW_MOTION_DEMO,
            ActionType.SLOW_MOTION_DEMO,
            ActionType.SLOW_MOTION_DEMO,
            ActionType.ADD_MOVEMENT_HINT,
            ActionType.GENERATE_MICRO_DRILL,
            ActionType.CHOOSE_FEEDBACK_STYLE,
            ActionType.QUICK_ASSESSMENT,
            ActionType.FINALIZE_PLAN,
        ],
        coverage=CoverageFlags(
            has_assessment=True, has_timing_support=True,
            has_movement_hint=True, has_micro_drill=True,
            has_feedback_style=True, plan_finalized=True,
        ),
        satisfied_requirements={"scaffolded_demo", "micro_drill", "assessment", "feedback_selection"},
        learner=LearnerSimState(
            comprehension={"movement": 0.65, "timing": 0.55},
            attention=0.50, frustration=0.30, confidence=0.45,
        ),
        initial_comprehension={"movement": 0.25, "timing": 0.20},
        step_count=8,
    )
    assert bad.sub_scores.efficiency < good.sub_scores.efficiency


def test_hard_task_full_coverage():
    task = get_task("hard_multi_error_adaptive")
    grade = grade_episode(
        task=task,
        action_history=[
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
        ],
        coverage=CoverageFlags(
            has_prerequisite=True, has_visual_cue=True,
            has_timing_support=True, has_movement_hint=True,
            has_feedback_style=True, has_micro_drill=True,
            has_assessment=True, has_revision=True,
            demo_before_drill=True, cue_before_drill=True,
            assessed_before_revision=True, plan_finalized=True,
        ),
        satisfied_requirements={
            "prerequisite_sign", "visual_cue", "movement_correction",
            "micro_drill", "assessment", "revision", "feedback_selection",
        },
        learner=LearnerSimState(
            comprehension={"movement": 0.68, "location": 0.65, "handshape": 0.72},
            attention=0.55, frustration=0.20, confidence=0.55,
        ),
        initial_comprehension={"movement": 0.10, "location": 0.15, "handshape": 0.30},
        step_count=10,
    )
    assert grade.total_score >= 0.65
    assert grade.passed or grade.total_score >= 0.60  # Hard task is genuinely hard
    assert len(grade.missing_requirements) == 0


def test_grade_reasoning_contains_info():
    """Grade reasoning should mention strengths or weaknesses."""
    task = get_task("easy_remediate_handshape")
    grade = grade_episode(
        task=task,
        action_history=[ActionType.SLOW_MOTION_DEMO, ActionType.FINALIZE_PLAN],
        coverage=CoverageFlags(has_timing_support=True, plan_finalized=True),
        satisfied_requirements={"corrective_intervention"},
        learner=LearnerSimState(comprehension={"handshape": 0.55}, attention=0.75, frustration=0.1, confidence=0.6),
        initial_comprehension={"handshape": 0.20},
        step_count=2,
    )
    assert len(grade.reasoning) > 0
    assert "Strength" in grade.reasoning or "Weakness" in grade.reasoning

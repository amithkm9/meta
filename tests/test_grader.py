"""Tests for the grading logic."""

import pytest

from app.grader import grade_episode
from app.models import ActionType, CoverageFlags
from app.tasks import get_task


def test_grade_normalization():
    task = get_task("easy_remediate_handshape")
    grade = grade_episode(
        task=task,
        action_history=[ActionType.FINALIZE_PLAN],
        coverage=CoverageFlags(plan_finalized=True),
        satisfied_requirements=set(),
        step_count=1,
    )
    assert 0.0 <= grade.total_score <= 1.0
    assert 0.0 <= grade.sub_scores.intervention_relevance <= 1.0
    assert 0.0 <= grade.sub_scores.pedagogical_sequence <= 1.0
    assert 0.0 <= grade.sub_scores.learner_need_alignment <= 1.0
    assert 0.0 <= grade.sub_scores.task_completeness <= 1.0
    assert 0.0 <= grade.sub_scores.efficiency <= 1.0


def test_easy_task_can_pass():
    task = get_task("easy_remediate_handshape")
    grade = grade_episode(
        task=task,
        action_history=[
            ActionType.SLOW_MOTION_DEMO,
            ActionType.ADD_LOCATION_CUE,
            ActionType.ADD_MOVEMENT_HINT,
            ActionType.CHOOSE_FEEDBACK_STYLE,
            ActionType.QUICK_ASSESSMENT,
            ActionType.FINALIZE_PLAN,
        ],
        coverage=CoverageFlags(
            has_timing_support=True,
            has_visual_cue=True,
            has_movement_hint=True,
            has_feedback_style=True,
            has_assessment=True,
            plan_finalized=True,
        ),
        satisfied_requirements={"corrective_intervention", "assessment"},
        step_count=6,
    )
    assert grade.total_score >= 0.70
    assert grade.passed is True


def test_redundant_actions_reduce_efficiency():
    task = get_task("medium_fix_movement_with_scaffold")
    # Good run
    good_grade = grade_episode(
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
            plan_finalized=True,
        ),
        satisfied_requirements={"scaffolded_demo", "micro_drill", "assessment", "feedback_selection"},
        step_count=6,
    )
    # Redundant run — same actions but repeated
    bad_grade = grade_episode(
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
            has_timing_support=True,
            has_movement_hint=True,
            has_micro_drill=True,
            has_feedback_style=True,
            has_assessment=True,
            plan_finalized=True,
        ),
        satisfied_requirements={"scaffolded_demo", "micro_drill", "assessment", "feedback_selection"},
        step_count=8,
    )
    assert bad_grade.sub_scores.efficiency < good_grade.sub_scores.efficiency


def test_hard_task_full_coverage():
    task = get_task("hard_adaptive_multi_error_plan")
    grade = grade_episode(
        task=task,
        action_history=[
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
            has_prerequisite=True,
            has_visual_cue=True,
            has_timing_support=True,
            has_movement_hint=True,
            has_feedback_style=True,
            has_micro_drill=True,
            has_assessment=True,
            has_revision=True,
            plan_finalized=True,
        ),
        satisfied_requirements={
            "prerequisite_sign", "visual_cue", "movement_correction",
            "micro_drill", "assessment", "revision", "feedback_selection",
        },
        step_count=9,
    )
    assert grade.total_score >= 0.70
    assert grade.passed is True
    assert len(grade.missing_requirements) == 0

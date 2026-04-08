"""Outcome-based step reward.

Rewards are computed from actual learner state changes, not just checklist coverage.
"""

from __future__ import annotations

from app.models import (
    ActionType,
    CoverageFlags,
    ErrorType,
    LearnerSimState,
    TaskSpec,
)

_NO_COMP_ACTIONS = {ActionType.QUICK_ASSESSMENT, ActionType.FINALIZE_PLAN}


def compute_step_reward(
    action: ActionType,
    task: TaskSpec,
    learner: LearnerSimState,
    comp_before: dict[str, float],
    comp_after: dict[str, float],
    coverage: CoverageFlags,
    action_history: list[ActionType],
    step: int,
) -> float:
    """Compute a single step reward in [0.0, 1.0]."""

    weights = task.grader_weights or {
        "comprehension_gain": 0.40,
        "pedagogical_quality": 0.25,
        "learner_wellbeing": 0.15,
        "efficiency": 0.20,
    }

    # 1. Comprehension gain (weighted by error severity)
    error_severity = {ep.error_type.value: ep.severity for ep in task.error_patterns}
    total_weighted_gain = 0.0
    total_weight = 0.0
    for et_val, sev in error_severity.items():
        before = comp_before.get(et_val, 0.0)
        after = comp_after.get(et_val, 0.0)
        total_weighted_gain += (after - before) * sev
        total_weight += sev

    if action in _NO_COMP_ACTIONS:
        # Assessment and finalize get moderate comp score (information value)
        comp_score = 0.4
    elif total_weight > 0:
        # Normalize: a gain of 0.15 weighted average is excellent
        comp_score = min(1.0, total_weighted_gain / total_weight / 0.15)
        comp_score = max(0.0, comp_score)
    else:
        comp_score = 0.0

    # 2. Pedagogical quality — sequence and relevance
    ideal = task.ideal_action_order
    ped_score = 0.5
    if action in ideal:
        ideal_idx = ideal.index(action)
        expected_frac = ideal_idx / max(len(ideal) - 1, 1)
        actual_frac = step / max(task.constraints.max_steps - 1, 1)
        diff = abs(expected_frac - actual_frac)
        ped_score = max(0.0, 1.0 - diff * 1.5)

    # Bonus for good sequencing patterns
    if action == ActionType.GENERATE_MICRO_DRILL and coverage.demo_before_drill:
        ped_score = min(1.0, ped_score + 0.15)
    if action == ActionType.REVISION_LOOP and coverage.assessed_before_revision:
        ped_score = min(1.0, ped_score + 0.15)

    # 3. Learner wellbeing — attention and frustration
    wellbeing = (learner.attention * 0.5 + (1.0 - learner.frustration) * 0.3 + learner.confidence * 0.2)
    wellbeing = min(1.0, max(0.0, wellbeing))

    # 4. Efficiency — penalize duplicates
    dup_count = action_history.count(action)
    if dup_count > 1:
        efficiency = max(0.0, 1.0 - (dup_count - 1) * 0.35)
    else:
        efficiency = 1.0

    # Weighted sum
    total = (
        weights.get("comprehension_gain", 0.40) * comp_score
        + weights.get("pedagogical_quality", 0.25) * ped_score
        + weights.get("learner_wellbeing", 0.15) * wellbeing
        + weights.get("efficiency", 0.20) * efficiency
    )
    return round(min(1.0, max(0.0, total)), 4)

from __future__ import annotations

from app.models import (
    ActionType,
    CoverageFlags,
    ErrorType,
    RewardBreakdown,
    TaskSpec,
)

# Maps action types to the requirement keywords they satisfy
_ACTION_TO_REQUIREMENT: dict[ActionType, list[str]] = {
    ActionType.SELECT_PREREQUISITE_SIGN: ["prerequisite_sign"],
    ActionType.SLOW_MOTION_DEMO: ["scaffolded_demo", "corrective_intervention"],
    ActionType.ADD_LOCATION_CUE: ["visual_cue", "corrective_intervention"],
    ActionType.ADD_MOVEMENT_HINT: ["movement_correction", "corrective_intervention"],
    ActionType.CHOOSE_FEEDBACK_STYLE: ["feedback_selection"],
    ActionType.GENERATE_MICRO_DRILL: ["micro_drill"],
    ActionType.QUICK_ASSESSMENT: ["assessment"],
    ActionType.REVISION_LOOP: ["revision"],
    ActionType.FINALIZE_PLAN: [],
}

# Maps action types to the error types they help address
_ACTION_RELEVANCE: dict[ActionType, list[ErrorType]] = {
    ActionType.SLOW_MOTION_DEMO: [ErrorType.MOVEMENT, ErrorType.TIMING],
    ActionType.ADD_LOCATION_CUE: [ErrorType.LOCATION],
    ActionType.ADD_MOVEMENT_HINT: [ErrorType.MOVEMENT, ErrorType.ORIENTATION],
    ActionType.GENERATE_MICRO_DRILL: [
        ErrorType.HANDSHAPE, ErrorType.MOVEMENT, ErrorType.LOCATION,
        ErrorType.TIMING, ErrorType.ORIENTATION,
    ],
    ActionType.CHOOSE_FEEDBACK_STYLE: [
        ErrorType.HANDSHAPE, ErrorType.MOVEMENT, ErrorType.LOCATION,
        ErrorType.TIMING, ErrorType.ORIENTATION,
    ],
}


def newly_satisfied_requirements(
    action: ActionType,
    already_satisfied: set[str],
    required: list[str],
) -> list[str]:
    """Return requirement keywords newly covered by this action."""
    covers = _ACTION_TO_REQUIREMENT.get(action, [])
    return [r for r in covers if r in required and r not in already_satisfied]


def compute_step_reward(
    action: ActionType,
    task: TaskSpec,
    coverage_before: CoverageFlags,
    coverage_after: CoverageFlags,
    action_history: list[ActionType],
    satisfied_before: set[str],
    satisfied_after: set[str],
    step: int,
) -> RewardBreakdown:
    """Compute intermediate reward for a single step."""
    weights = task.grader_weights or {
        "intervention_relevance": 0.25,
        "pedagogical_sequence": 0.25,
        "learner_need_alignment": 0.20,
        "task_completeness": 0.20,
        "efficiency": 0.10,
    }

    # 1. Intervention relevance – does the action address an active error?
    error_types = {ep.error_type for ep in task.error_patterns}
    relevant_errors = _ACTION_RELEVANCE.get(action, [])
    if action == ActionType.FINALIZE_PLAN:
        relevance = 0.5
    elif action in (ActionType.SELECT_PREREQUISITE_SIGN, ActionType.QUICK_ASSESSMENT, ActionType.REVISION_LOOP):
        relevance = 0.7
    elif any(e in error_types for e in relevant_errors):
        relevance = 1.0
    else:
        relevance = 0.2

    # 2. Pedagogical sequence – is the action in a sensible order?
    ideal = task.ideal_action_order
    seq_score = 0.5  # default
    if action in ideal:
        ideal_idx = ideal.index(action)
        # Reward if it's roughly in the right position
        expected_fraction = ideal_idx / max(len(ideal) - 1, 1)
        actual_fraction = step / max(task.constraints.max_steps - 1, 1)
        diff = abs(expected_fraction - actual_fraction)
        seq_score = max(0.0, 1.0 - diff * 2)

    # 3. Learner need alignment – does the action match support needs?
    needs = {n.value for n in task.learner.support_needs}
    need_score = 0.5
    if action == ActionType.SLOW_MOTION_DEMO and "slowed_pacing" in needs:
        need_score = 1.0
    elif action == ActionType.ADD_LOCATION_CUE and "visual_aids" in needs:
        need_score = 1.0
    elif action == ActionType.CHOOSE_FEEDBACK_STYLE:
        need_score = 0.8
    elif action == ActionType.REVISION_LOOP and "repetition" in needs:
        need_score = 1.0
    elif action == ActionType.FINALIZE_PLAN:
        need_score = 0.5

    # 4. Task completeness – fraction of requirements satisfied
    total_req = len(task.constraints.required_outputs)
    completeness = len(satisfied_after) / max(total_req, 1)

    # 5. Efficiency – penalise duplicate actions
    dup_count = action_history.count(action)
    if dup_count > 1:
        efficiency = max(0.0, 1.0 - (dup_count - 1) * 0.3)
    else:
        efficiency = 1.0

    # Weighted sum
    total = (
        weights.get("intervention_relevance", 0.25) * relevance
        + weights.get("pedagogical_sequence", 0.25) * seq_score
        + weights.get("learner_need_alignment", 0.20) * need_score
        + weights.get("task_completeness", 0.20) * completeness
        + weights.get("efficiency", 0.10) * efficiency
    )
    total = round(min(1.0, max(0.0, total)), 4)

    return RewardBreakdown(
        intervention_relevance=round(relevance, 4),
        pedagogical_sequence=round(seq_score, 4),
        learner_need_alignment=round(need_score, 4),
        task_completeness=round(completeness, 4),
        efficiency=round(efficiency, 4),
        total=total,
    )

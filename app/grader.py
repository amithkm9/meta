"""Outcome-based final grading.

The final grade primarily measures actual learner comprehension improvement,
not just whether the agent ticked checklist items.
"""

from __future__ import annotations

from app.models import (
    ActionType,
    CoverageFlags,
    GradeReport,
    LearnerSimState,
    RewardBreakdown,
    TaskSpec,
)


def grade_episode(
    task: TaskSpec,
    action_history: list[ActionType],
    coverage: CoverageFlags,
    satisfied_requirements: set[str],
    learner: LearnerSimState,
    initial_comprehension: dict[str, float],
    step_count: int,
) -> GradeReport:
    """Deterministic final grading based on learner outcomes. Score in [0,1]."""

    weights = task.grader_weights or {
        "comprehension_gain": 0.40,
        "pedagogical_quality": 0.25,
        "learner_wellbeing": 0.15,
        "efficiency": 0.20,
    }

    error_severity = {ep.error_type.value: ep.severity for ep in task.error_patterns}
    target = task.constraints.comprehension_target

    # ── 1. Comprehension gain (primary metric) ─────────────────────────
    # How much did the learner actually improve, weighted by error severity?
    total_weight = sum(error_severity.values()) or 1.0
    weighted_improvement = 0.0
    errors_meeting_target = 0
    total_errors = len(error_severity)

    for et_val, sev in error_severity.items():
        before = initial_comprehension.get(et_val, 0.0)
        after = learner.comprehension.get(et_val, 0.0)
        # Improvement relative to what was needed
        needed = max(0.01, target - before)
        achieved = after - before
        ratio = min(1.0, achieved / needed) if needed > 0 else 1.0
        weighted_improvement += ratio * sev
        if after >= target:
            errors_meeting_target += 1

    comp_score = weighted_improvement / total_weight
    # Bonus for meeting target on all errors
    target_ratio = errors_meeting_target / max(total_errors, 1)
    comp_score = comp_score * 0.7 + target_ratio * 0.3
    comp_score = min(1.0, max(0.0, comp_score))

    # ── 2. Pedagogical quality ─────────────────────────────────────────
    # Sequence ordering (Kendall-tau-like) + requirement coverage
    ideal = task.ideal_action_order
    ideal_pos = {a: i for i, a in enumerate(ideal)}
    filtered = [a for a in action_history if a in ideal_pos]

    if len(filtered) >= 2:
        concordant = 0
        total_pairs = 0
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                total_pairs += 1
                if ideal_pos[filtered[i]] < ideal_pos[filtered[j]]:
                    concordant += 1
        seq_score = concordant / max(total_pairs, 1)
    elif len(filtered) == 1:
        seq_score = 0.6
    else:
        seq_score = 0.0

    # Requirement coverage
    required = set(task.constraints.required_outputs)
    req_coverage = len(satisfied_requirements & required) / max(len(required), 1)

    # Sequence bonus patterns
    pattern_bonus = 0.0
    if coverage.demo_before_drill:
        pattern_bonus += 0.1
    if coverage.assessed_before_revision:
        pattern_bonus += 0.1
    if coverage.cue_before_drill:
        pattern_bonus += 0.05

    ped_score = seq_score * 0.4 + req_coverage * 0.4 + min(0.2, pattern_bonus)
    ped_score = min(1.0, max(0.0, ped_score))

    # ── 3. Learner wellbeing ───────────────────────────────────────────
    # Did the agent keep the learner engaged and not frustrated?
    wellbeing = (
        learner.attention * 0.35
        + (1.0 - learner.frustration) * 0.35
        + learner.confidence * 0.30
    )
    wellbeing = min(1.0, max(0.0, wellbeing))

    # ── 4. Efficiency ──────────────────────────────────────────────────
    # Fewer steps = better, penalize duplicates
    unique_actions = len(set(action_history))
    dup_penalty = max(0.0, 1.0 - (len(action_history) - unique_actions) * 0.12)
    step_ratio = max(0.0, 1.0 - step_count / max(task.constraints.max_steps, 1) * 0.25)
    efficiency = (step_ratio + dup_penalty) / 2
    efficiency = min(1.0, max(0.0, efficiency))

    # ── Weighted total ─────────────────────────────────────────────────
    total = (
        weights.get("comprehension_gain", 0.40) * comp_score
        + weights.get("pedagogical_quality", 0.25) * ped_score
        + weights.get("learner_wellbeing", 0.15) * wellbeing
        + weights.get("efficiency", 0.20) * efficiency
    )
    total = round(min(1.0, max(0.0, total)), 4)

    # ── Reasoning ──────────────────────────────────────────────────────
    missing = sorted(required - satisfied_requirements)
    strengths, weaknesses = [], []

    if comp_score >= 0.7:
        strengths.append(f"Strong learner improvement: {errors_meeting_target}/{total_errors} errors reached target comprehension.")
    elif comp_score < 0.4:
        weaknesses.append(f"Learner comprehension barely improved — only {errors_meeting_target}/{total_errors} errors reached target.")

    if ped_score >= 0.7:
        strengths.append("Good pedagogical sequencing and requirement coverage.")
    if coverage.demo_before_drill:
        strengths.append("Effective: demonstrated before drilling.")
    if coverage.assessed_before_revision:
        strengths.append("Smart: assessed before revising.")

    if wellbeing >= 0.65:
        strengths.append("Learner remained engaged and not frustrated.")
    elif learner.frustration > 0.5:
        weaknesses.append("Learner became frustrated — needed more encouragement or varied approach.")
    elif learner.attention < 0.35:
        weaknesses.append("Learner attention dropped critically — needed more engaging interventions.")

    if efficiency < 0.5:
        weaknesses.append("Too many redundant actions wasted the step budget.")

    if missing:
        weaknesses.append(f"Missing requirements: {', '.join(missing)}.")

    reasoning = ""
    if strengths:
        reasoning += "Strengths: " + " ".join(strengths)
    if weaknesses:
        reasoning += " Weaknesses: " + " ".join(weaknesses)
    if not reasoning:
        reasoning = "Average performance across dimensions."

    return GradeReport(
        total_score=total,
        sub_scores=RewardBreakdown(
            comprehension_gain=round(comp_score, 4),
            pedagogical_quality=round(ped_score, 4),
            learner_wellbeing=round(wellbeing, 4),
            efficiency=round(efficiency, 4),
            total=total,
        ),
        passed=total >= 0.70,
        reasoning=reasoning.strip(),
        comprehension_before={k: round(v, 2) for k, v in initial_comprehension.items()},
        comprehension_after={k: round(v, 2) for k, v in learner.comprehension.items()},
        missing_requirements=missing,
    )

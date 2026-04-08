from __future__ import annotations

from app.models import (
    ActionType,
    CoverageFlags,
    GradeReport,
    RewardBreakdown,
    TaskSpec,
)


def grade_episode(
    task: TaskSpec,
    action_history: list[ActionType],
    coverage: CoverageFlags,
    satisfied_requirements: set[str],
    step_count: int,
) -> GradeReport:
    """Deterministic final grading. Returns a GradeReport with score in [0,1]."""
    weights = task.grader_weights or {
        "intervention_relevance": 0.25,
        "pedagogical_sequence": 0.25,
        "learner_need_alignment": 0.20,
        "task_completeness": 0.20,
        "efficiency": 0.10,
    }

    required = set(task.constraints.required_outputs)
    missing = sorted(required - satisfied_requirements)

    # 1. Task completeness
    completeness = len(satisfied_requirements & required) / max(len(required), 1)

    # 2. Intervention relevance – what fraction of actions address known errors?
    from app.reward import _ACTION_RELEVANCE
    error_types = {ep.error_type for ep in task.error_patterns}
    relevant_count = 0
    for a in action_history:
        if a in (ActionType.FINALIZE_PLAN, ActionType.QUICK_ASSESSMENT,
                 ActionType.REVISION_LOOP, ActionType.SELECT_PREREQUISITE_SIGN):
            relevant_count += 1
        elif any(e in error_types for e in _ACTION_RELEVANCE.get(a, [])):
            relevant_count += 1
    relevance = relevant_count / max(len(action_history), 1)

    # 3. Pedagogical sequence – Kendall-tau-like ordering score
    ideal = task.ideal_action_order
    # Build order mapping from ideal
    ideal_pos = {a: i for i, a in enumerate(ideal)}
    # Filter action_history to only those in ideal
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
        seq_score = 0.7
    else:
        seq_score = 0.0

    # 4. Learner need alignment
    needs = {n.value for n in task.learner.support_needs}
    need_hits = 0
    need_checks = 0
    if "slowed_pacing" in needs:
        need_checks += 1
        if coverage.has_timing_support:
            need_hits += 1
    if "visual_aids" in needs:
        need_checks += 1
        if coverage.has_visual_cue:
            need_hits += 1
    if "repetition" in needs:
        need_checks += 1
        if coverage.has_revision or coverage.has_micro_drill:
            need_hits += 1
    if "attention_support" in needs:
        need_checks += 1
        if coverage.has_feedback_style:
            need_hits += 1
    if "tactile_guidance" in needs:
        need_checks += 1
        if coverage.has_feedback_style:
            need_hits += 1
    need_score = need_hits / max(need_checks, 1) if need_checks > 0 else 0.5

    # 5. Efficiency – fewer steps relative to max is better, penalize duplicates
    unique_actions = len(set(action_history))
    dup_penalty = 1.0 - (len(action_history) - unique_actions) * 0.1
    dup_penalty = max(0.0, dup_penalty)
    step_ratio = 1.0 - (step_count / max(task.constraints.max_steps, 1)) * 0.3
    efficiency = min(1.0, max(0.0, (step_ratio + dup_penalty) / 2))

    # Weighted total
    total = (
        weights.get("intervention_relevance", 0.25) * relevance
        + weights.get("pedagogical_sequence", 0.25) * seq_score
        + weights.get("learner_need_alignment", 0.20) * need_score
        + weights.get("task_completeness", 0.20) * completeness
        + weights.get("efficiency", 0.10) * efficiency
    )
    total = round(min(1.0, max(0.0, total)), 4)

    # Reasoning
    strengths = []
    weaknesses = []
    if completeness >= 0.8:
        strengths.append("Covered most required outputs.")
    elif completeness < 0.5:
        weaknesses.append("Many required outputs missing.")
    if seq_score >= 0.7:
        strengths.append("Good pedagogical ordering.")
    elif seq_score < 0.4:
        weaknesses.append("Action order deviates significantly from ideal sequence.")
    if relevance >= 0.7:
        strengths.append("Interventions well-targeted to error patterns.")
    if efficiency < 0.5:
        weaknesses.append("Too many redundant or wasted steps.")
    if need_score >= 0.7:
        strengths.append("Good alignment with learner support needs.")

    reasoning = ""
    if strengths:
        reasoning += "Strengths: " + " ".join(strengths)
    if weaknesses:
        reasoning += " Weaknesses: " + " ".join(weaknesses)
    if not reasoning:
        reasoning = "Average performance across all dimensions."

    return GradeReport(
        total_score=total,
        sub_scores=RewardBreakdown(
            intervention_relevance=round(relevance, 4),
            pedagogical_sequence=round(seq_score, 4),
            learner_need_alignment=round(need_score, 4),
            task_completeness=round(completeness, 4),
            efficiency=round(efficiency, 4),
            total=total,
        ),
        passed=total >= 0.70,
        reasoning=reasoning.strip(),
        missing_requirements=missing,
    )

"""Core tutoring environment with dynamic learner simulation.

The learner has hidden internal state (comprehension, attention, frustration,
confidence) that evolves probabilistically in response to the agent's
interventions. The agent must assess to reveal comprehension, read learner
signals, and adapt its strategy.
"""

from __future__ import annotations

import random as _random
import uuid
from typing import Any

from app.grader import grade_episode
from app.models import (
    Action,
    ActionType,
    CoverageFlags,
    ErrorType,
    GradeReport,
    LearnerSignals,
    LearnerSimState,
    LearningStyle,
    Observation,
    State,
    TaskSpec,
)
from app.reward import compute_step_reward
from app.tasks import default_task_id, get_task

# Optional PyTorch learner model (blended with hand-crafted sim)
# Disable with SIGNADAPT_NO_NEURAL=1 (e.g., during tests or low-resource deployments)
import os as _os
_USE_NEURAL_MODEL = _os.getenv("SIGNADAPT_NO_NEURAL", "").strip() not in ("1", "true")
try:
    from app.learner_model import get_learner_model, predict_gains, LearnerBehaviorNet
except ImportError:
    _USE_NEURAL_MODEL = False

# ── Learning style modifiers ─────────────────────────────────────────────
# learning_style → action_type → effectiveness multiplier
_LEARNING_STYLE_MODS: dict[LearningStyle, dict[ActionType, float]] = {
    LearningStyle.VISUAL: {
        ActionType.ADD_LOCATION_CUE: 1.30,
        ActionType.SLOW_MOTION_DEMO: 1.20,
        ActionType.ADD_MOVEMENT_HINT: 1.10,
        ActionType.GENERATE_MICRO_DRILL: 0.90,
    },
    LearningStyle.KINESTHETIC: {
        ActionType.GENERATE_MICRO_DRILL: 1.30,
        ActionType.REVISION_LOOP: 1.20,
        ActionType.ADD_MOVEMENT_HINT: 1.15,
        ActionType.SLOW_MOTION_DEMO: 0.90,
    },
    LearningStyle.AUDITORY: {
        ActionType.CHOOSE_FEEDBACK_STYLE: 1.25,
        ActionType.SELECT_PREREQUISITE_SIGN: 1.15,
        ActionType.SLOW_MOTION_DEMO: 0.85,
    },
    LearningStyle.MIXED: {},  # No modifiers
}

# ── Intervention effectiveness matrix ──────────────────────────────────
# base_effect[action_type][error_type] → base comprehension gain
# These are before modifiers (attention, frustration, sequence bonuses)

_BASE_EFFECT: dict[ActionType, dict[ErrorType, float]] = {
    ActionType.SELECT_PREREQUISITE_SIGN: {
        ErrorType.HANDSHAPE: 0.06, ErrorType.MOVEMENT: 0.06,
        ErrorType.LOCATION: 0.06, ErrorType.TIMING: 0.06,
        ErrorType.ORIENTATION: 0.06,
    },
    ActionType.SLOW_MOTION_DEMO: {
        ErrorType.HANDSHAPE: 0.10, ErrorType.MOVEMENT: 0.22,
        ErrorType.LOCATION: 0.05, ErrorType.TIMING: 0.20,
        ErrorType.ORIENTATION: 0.08,
    },
    ActionType.ADD_LOCATION_CUE: {
        ErrorType.HANDSHAPE: 0.02, ErrorType.MOVEMENT: 0.04,
        ErrorType.LOCATION: 0.22, ErrorType.TIMING: 0.02,
        ErrorType.ORIENTATION: 0.05,
    },
    ActionType.ADD_MOVEMENT_HINT: {
        ErrorType.HANDSHAPE: 0.03, ErrorType.MOVEMENT: 0.18,
        ErrorType.LOCATION: 0.04, ErrorType.TIMING: 0.06,
        ErrorType.ORIENTATION: 0.12,
    },
    ActionType.CHOOSE_FEEDBACK_STYLE: {
        ErrorType.HANDSHAPE: 0.05, ErrorType.MOVEMENT: 0.05,
        ErrorType.LOCATION: 0.05, ErrorType.TIMING: 0.05,
        ErrorType.ORIENTATION: 0.05,
    },
    ActionType.GENERATE_MICRO_DRILL: {
        ErrorType.HANDSHAPE: 0.12, ErrorType.MOVEMENT: 0.14,
        ErrorType.LOCATION: 0.12, ErrorType.TIMING: 0.14,
        ErrorType.ORIENTATION: 0.10,
    },
    ActionType.REVISION_LOOP: {
        ErrorType.HANDSHAPE: 0.08, ErrorType.MOVEMENT: 0.10,
        ErrorType.LOCATION: 0.08, ErrorType.TIMING: 0.10,
        ErrorType.ORIENTATION: 0.08,
    },
}

# Actions with no direct comprehension effect
_NO_EFFECT_ACTIONS = {ActionType.QUICK_ASSESSMENT, ActionType.FINALIZE_PLAN}


class TutoringEnv:
    """Tutoring environment with a dynamic learner simulation."""

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task: TaskSpec | None = None
        self._step_count: int = 0
        self._done: bool = True
        self._action_history: list[ActionType] = []
        self._plan: list[str] = []
        self._coverage: CoverageFlags = CoverageFlags()
        self._satisfied: set[str] = set()
        self._cumulative_reward: float = 0.0
        self._last_step_reward: float | None = None
        self._final_grade: GradeReport | None = None
        # Simulation
        self._learner: LearnerSimState = LearnerSimState()
        self._initial_comprehension: dict[str, float] = {}
        self._rng: _random.Random = _random.Random(42)
        self._signals: LearnerSignals = LearnerSignals()
        self._steps_since_assessment: int = 999
        self._last_assessed_comprehension: dict[str, float] | None = None
        self._fatigue: float = 0.0  # accumulated fatigue (0..1)

    # ── public interface ───────────────────────────────────────────────

    def reset(self, task_id: str | None = None, episode_id: str | None = None, seed: int | None = None) -> Observation:
        tid = task_id or default_task_id()
        task = get_task(tid)
        self._task = task
        self._episode_id = episode_id or uuid.uuid4().hex[:12]
        self._step_count = 0
        self._done = False
        self._action_history = []
        self._plan = []
        self._coverage = CoverageFlags()
        self._satisfied = set()
        self._cumulative_reward = 0.0
        self._last_step_reward = None
        self._final_grade = None
        self._steps_since_assessment = 999
        self._last_assessed_comprehension = None
        self._fatigue = 0.0

        # Init simulation
        sim = task.simulation
        actual_seed = seed if seed is not None else sim.default_seed
        self._rng = _random.Random(actual_seed)
        self._learner = LearnerSimState(
            comprehension=dict(sim.initial_comprehension),
            attention=sim.initial_attention,
            frustration=sim.initial_frustration,
            confidence=sim.initial_confidence,
        )
        self._initial_comprehension = dict(sim.initial_comprehension)
        self._signals = self._make_signals("Episode started. Observe the learner and plan your approach.")
        return self._observation()

    def step(self, action: Action) -> Observation:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        task = self._task
        assert task is not None

        # Snapshot before
        comp_before = dict(self._learner.comprehension)

        # Apply simulation effects
        self._simulate_action(action, comp_before)
        self._step_count += 1
        self._steps_since_assessment += 1
        self._action_history.append(action.action_type)

        # Track requirement satisfaction
        self._update_requirements(action.action_type)

        # Attention decay
        self._learner.attention = max(0.0, self._learner.attention - task.simulation.attention_decay_per_step)

        # Memory decay — comprehension fades slightly each step if not reinforced
        decay_rate = task.simulation.memory_decay_rate
        if decay_rate > 0:
            for et_val in list(self._learner.comprehension):
                old_val = self._learner.comprehension[et_val]
                # Don't decay below initial comprehension
                floor = self._initial_comprehension.get(et_val, 0.0)
                decayed = old_val - decay_rate * self._rng.uniform(0.5, 1.5)
                self._learner.comprehension[et_val] = round(max(floor, decayed), 4)

        # Fatigue accumulation
        fatigue_rate = task.simulation.fatigue_rate
        if fatigue_rate > 0:
            self._fatigue = min(1.0, self._fatigue + fatigue_rate)

        # Compute reward
        comp_after = dict(self._learner.comprehension)
        reward = compute_step_reward(
            action=action.action_type,
            task=task,
            learner=self._learner,
            comp_before=comp_before,
            comp_after=comp_after,
            coverage=self._coverage,
            action_history=self._action_history,
            step=self._step_count - 1,
        )
        self._last_step_reward = reward
        self._cumulative_reward += reward

        # Termination
        if action.action_type == ActionType.FINALIZE_PLAN:
            self._done = True
        elif self._step_count >= task.constraints.max_steps:
            self._done = True

        if self._done:
            self._final_grade = grade_episode(
                task=task,
                action_history=self._action_history,
                coverage=self._coverage,
                satisfied_requirements=self._satisfied,
                learner=self._learner,
                initial_comprehension=self._initial_comprehension,
                step_count=self._step_count,
            )

        return self._observation()

    @property
    def state(self) -> State:
        task = self._task
        return State(
            episode_id=self._episode_id or None,
            step_count=self._step_count,
            task_id=task.id if task else "",
            max_steps=task.constraints.max_steps if task else 0,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            coverage=self._coverage,
            current_plan=list(self._plan),
            final_grade=self._final_grade,
        )

    @property
    def last_step_reward(self) -> float | None:
        return self._last_step_reward

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def final_grade(self) -> GradeReport | None:
        return self._final_grade

    # ── simulation engine ──────────────────────────────────────────────

    def _simulate_action(self, action: Action, comp_before: dict[str, float] | None = None) -> None:
        at = action.action_type
        task = self._task
        assert task is not None
        L = self._learner
        error_types = [ep.error_type for ep in task.error_patterns]
        if comp_before is None:
            comp_before = dict(L.comprehension)

        if at == ActionType.FINALIZE_PLAN:
            self._coverage.plan_finalized = True
            self._plan.append(f"Plan finalized: {action.rationale or 'complete'}")
            self._signals = self._make_signals("Tutoring plan submitted for review.")
            return

        if at == ActionType.QUICK_ASSESSMENT:
            # Assessment reveals hidden comprehension
            self._coverage.has_assessment = True
            self._steps_since_assessment = 0
            self._last_assessed_comprehension = dict(L.comprehension)
            # Mild stress cost
            L.frustration = min(1.0, L.frustration + 0.03)
            L.attention = max(0.0, L.attention - 0.02)
            self._plan.append("Assessment: checked learner comprehension levels.")
            self._signals = self._make_signals(
                "Assessment complete. Comprehension levels revealed.",
                assessed=True,
            )
            return

        # ── Compute comprehension gains for teaching actions ───────────
        # Sequence bonuses
        seq_mult = 1.0
        if self._coverage.has_prerequisite:
            seq_mult *= 1.25
        if at == ActionType.GENERATE_MICRO_DRILL:
            if self._coverage.has_timing_support:  # demo was given
                self._coverage.demo_before_drill = True
                seq_mult *= 1.30
            if self._coverage.has_visual_cue or self._coverage.has_movement_hint:
                self._coverage.cue_before_drill = True
                seq_mult *= 1.15
        if at == ActionType.REVISION_LOOP:
            if self._steps_since_assessment <= 2:
                self._coverage.assessed_before_revision = True
                seq_mult *= 1.25  # targeted revision

        # Support need match bonus
        needs = {n.value for n in task.learner.support_needs}
        support_mult = 1.0
        if at == ActionType.SLOW_MOTION_DEMO and "slowed_pacing" in needs:
            support_mult = 1.20
        elif at == ActionType.ADD_LOCATION_CUE and "visual_aids" in needs:
            support_mult = 1.20
        elif at == ActionType.REVISION_LOOP and "repetition" in needs:
            support_mult = 1.15
        elif at == ActionType.CHOOSE_FEEDBACK_STYLE and "attention_support" in needs:
            support_mult = 1.15

        # Learning style modifier
        style = task.simulation.learning_style
        style_mods = _LEARNING_STYLE_MODS.get(style, {})
        style_mult = style_mods.get(at, 1.0)

        # Fatigue reduces effectiveness
        fatigue_penalty = 1.0 - self._fatigue * 0.35

        # Apply effect per error
        effects = _BASE_EFFECT.get(at, {})
        total_gain = 0.0
        for et in error_types:
            base = effects.get(et, 0.02)
            noise = self._rng.uniform(0.80, 1.20)
            frustration_penalty = 1.0 - L.frustration * 0.4
            modifier = L.attention * frustration_penalty * seq_mult * support_mult * style_mult * fatigue_penalty * noise
            gain = base * modifier
            old_comp = L.comprehension.get(et.value, 0.0)
            # Diminishing returns near 1.0
            effective_gain = gain * (1.0 - old_comp * 0.3)
            new_comp = min(1.0, old_comp + effective_gain)
            L.comprehension[et.value] = round(new_comp, 4)
            total_gain += new_comp - old_comp

        # Blend with PyTorch neural model predictions (20% neural, 80% hand-crafted)
        if _USE_NEURAL_MODEL and at not in _NO_EFFECT_ACTIONS:
            try:
                model = get_learner_model()
                style_val = task.simulation.learning_style
                style_str = style_val.value if hasattr(style_val, 'value') else str(style_val)
                preds = predict_gains(
                    model=model,
                    action_type=at.value,
                    comprehension=comp_before,
                    attention=L.attention,
                    frustration=L.frustration,
                    confidence=L.confidence,
                    fatigue=self._fatigue,
                    step=self._step_count,
                    max_steps=task.constraints.max_steps,
                    has_assessed=self._coverage.has_assessment,
                    has_demo=self._coverage.has_timing_support,
                    has_drill=self._coverage.has_micro_drill,
                    has_feedback=self._coverage.has_feedback_style,
                    has_prerequisite=self._coverage.has_prerequisite,
                    learning_style=style_str,
                )
                # Blend: 80% hand-crafted + 20% neural
                neural_weight = 0.20
                for et in error_types:
                    neural_gain = preds["comprehension_gains"].get(et.value, 0.0)
                    hc_val = L.comprehension.get(et.value, 0.0)
                    old_val = comp_before.get(et.value, 0.0)
                    hc_gain = hc_val - old_val
                    blended = hc_gain * (1 - neural_weight) + neural_gain * neural_weight
                    L.comprehension[et.value] = round(min(1.0, old_val + max(0.0, blended)), 4)
                # Recalculate total_gain after blending
                total_gain = sum(
                    L.comprehension.get(et.value, 0.0) - comp_before.get(et.value, 0.0)
                    for et in error_types
                )
            except Exception:
                pass  # Fall back to pure hand-crafted gains

        # Update learner emotional state based on outcome
        if total_gain > 0.08:
            L.confidence = min(1.0, L.confidence + 0.05)
            L.frustration = max(0.0, L.frustration - 0.03)
        elif total_gain < 0.02:
            L.frustration = min(1.0, L.frustration + 0.04)
            L.confidence = max(0.0, L.confidence - 0.02)

        # Action-specific side effects
        if at == ActionType.SELECT_PREREQUISITE_SIGN:
            self._coverage.has_prerequisite = True
            L.attention = min(1.0, L.attention + 0.03)
            sign = action.payload.get("sign", "foundational-sign")
            self._plan.append(f"Prerequisite: {sign}")
        elif at == ActionType.SLOW_MOTION_DEMO:
            self._coverage.has_timing_support = True
            L.attention = min(1.0, L.attention + 0.06)
            self._plan.append("Slow-motion demonstration.")
        elif at == ActionType.ADD_LOCATION_CUE:
            self._coverage.has_visual_cue = True
            cue = action.payload.get("cue", "spatial marker")
            L.attention = min(1.0, L.attention + 0.03)
            self._plan.append(f"Location cue: {cue}")
        elif at == ActionType.ADD_MOVEMENT_HINT:
            self._coverage.has_movement_hint = True
            hint = action.payload.get("hint", "directional overlay")
            self._plan.append(f"Movement hint: {hint}")
        elif at == ActionType.CHOOSE_FEEDBACK_STYLE:
            self._coverage.has_feedback_style = True
            L.frustration = max(0.0, L.frustration - 0.08)
            L.attention = min(1.0, L.attention + 0.04)
            style = action.payload.get("style", "visual")
            self._plan.append(f"Feedback: {style}")
        elif at == ActionType.GENERATE_MICRO_DRILL:
            self._coverage.has_micro_drill = True
            L.attention = max(0.0, L.attention - 0.04)
            focus = action.payload.get("focus", "target sign")
            self._plan.append(f"Micro-drill: {focus}")
        elif at == ActionType.REVISION_LOOP:
            self._coverage.has_revision = True
            L.attention = max(0.0, L.attention - 0.06)
            self._plan.append("Revision loop.")

        # Duplicate action penalty — repeated same action has reduced novelty
        dup_count = self._action_history.count(at)
        if dup_count >= 2:
            L.frustration = min(1.0, L.frustration + 0.05)
            L.attention = max(0.0, L.attention - 0.03)

        # Generate signals
        if total_gain > 0.12:
            feedback = "Learner shows clear improvement — eyes bright, hands more precise."
        elif total_gain > 0.05:
            feedback = "Learner shows some progress — movements becoming more intentional."
        elif total_gain > 0.01:
            feedback = "Intervention had minimal visible effect."
        else:
            feedback = "Learner seems unchanged or slightly confused after intervention."
        self._signals = self._make_signals(feedback)

    def _update_requirements(self, at: ActionType) -> None:
        """Track which grading requirements have been satisfied."""
        req_map: dict[ActionType, list[str]] = {
            ActionType.SELECT_PREREQUISITE_SIGN: ["prerequisite_sign"],
            ActionType.SLOW_MOTION_DEMO: ["scaffolded_demo", "corrective_intervention"],
            ActionType.ADD_LOCATION_CUE: ["visual_cue", "corrective_intervention"],
            ActionType.ADD_MOVEMENT_HINT: ["movement_correction", "corrective_intervention"],
            ActionType.CHOOSE_FEEDBACK_STYLE: ["feedback_selection"],
            ActionType.GENERATE_MICRO_DRILL: ["micro_drill"],
            ActionType.QUICK_ASSESSMENT: ["assessment"],
            ActionType.REVISION_LOOP: ["revision"],
        }
        task = self._task
        assert task is not None
        required = set(task.constraints.required_outputs)
        for r in req_map.get(at, []):
            if r in required:
                self._satisfied.add(r)

    def _make_signals(self, feedback: str, assessed: bool = False) -> LearnerSignals:
        L = self._learner
        # Engagement
        if L.attention > 0.75:
            engagement = "Learner is attentive and engaged."
        elif L.attention > 0.50:
            engagement = "Learner seems somewhat distracted."
        elif L.attention > 0.30:
            engagement = "Learner is losing focus — consider a more engaging approach."
        else:
            engagement = "Learner has largely disengaged."

        # Emotional state
        if L.frustration > 0.6:
            emotional = "Learner shows visible frustration — consider encouragement or a different tactic."
        elif L.frustration > 0.35:
            emotional = "Learner is slightly stressed but coping."
        elif L.confidence > 0.6:
            emotional = "Learner appears confident and receptive."
        else:
            emotional = "Learner is calm but uncertain."

        assessed_comp = None
        if assessed or self._steps_since_assessment <= 1:
            assessed_comp = {k: round(v, 2) for k, v in L.comprehension.items()}

        return LearnerSignals(
            engagement=engagement,
            emotional_state=emotional,
            intervention_feedback=feedback,
            assessed_comprehension=assessed_comp,
        )

    # ── observation builder ────────────────────────────────────────────

    def _observation(self) -> Observation:
        task = self._task
        assert task is not None
        return Observation(
            done=self._done,
            reward=self._last_step_reward,
            metadata={
                "episode_id": self._episode_id,
                "final_grade": self._final_grade.model_dump() if self._final_grade else None,
            },
            task_id=task.id,
            difficulty=task.difficulty,
            learner=task.learner,
            lesson_goal=task.lesson_goal,
            error_patterns=task.error_patterns,
            support_needs=task.learner.support_needs,
            current_plan=list(self._plan),
            completed_requirements=sorted(self._satisfied),
            remaining_steps=task.constraints.max_steps - self._step_count,
            allowed_actions=list(ActionType),
            coverage=self._coverage,
            learner_signals=self._signals,
            step_count=self._step_count,
        )

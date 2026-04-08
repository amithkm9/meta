from __future__ import annotations

import uuid

from app.grader import grade_episode
from app.models import (
    Action,
    ActionType,
    CoverageFlags,
    GradeReport,
    Observation,
    State,
    TaskSpec,
)
from app.reward import compute_step_reward, newly_satisfied_requirements
from app.tasks import default_task_id, get_task


class TutoringEnv:
    """Stateful tutoring environment. One active episode at a time."""

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
        self._last_action_result: str | None = None
        self._final_grade: GradeReport | None = None
        self._last_step_reward: float | None = None

    # ── public interface ───────────────────────────────────────────────

    def reset(self, task_id: str | None = None, episode_id: str | None = None) -> Observation:
        """Reset and return the initial Observation (OpenEnv-compliant)."""
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
        self._last_action_result = None
        self._final_grade = None
        self._last_step_reward = None
        return self._observation()

    def step(self, action: Action) -> Observation:
        """Process action and return the resulting Observation (OpenEnv-compliant)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        task = self._task
        assert task is not None

        coverage_before = self._coverage.model_copy()
        satisfied_before = set(self._satisfied)

        # Apply action effects
        self._apply_action(action)
        self._step_count += 1
        self._action_history.append(action.action_type)

        # Check newly satisfied requirements
        new_reqs = newly_satisfied_requirements(
            action.action_type, satisfied_before, task.constraints.required_outputs,
        )
        self._satisfied.update(new_reqs)

        # Compute reward
        reward_bd = compute_step_reward(
            action=action.action_type,
            task=task,
            coverage_before=coverage_before,
            coverage_after=self._coverage,
            action_history=self._action_history,
            satisfied_before=satisfied_before,
            satisfied_after=self._satisfied,
            step=self._step_count - 1,
        )
        step_reward = reward_bd.total
        self._cumulative_reward += step_reward
        self._last_step_reward = step_reward

        # Check termination
        if action.action_type == ActionType.FINALIZE_PLAN:
            self._done = True
        elif self._step_count >= task.constraints.max_steps:
            self._done = True
            self._last_action_result = "Max steps reached. Auto-finalizing."

        if self._done:
            self._final_grade = grade_episode(
                task=task,
                action_history=self._action_history,
                coverage=self._coverage,
                satisfied_requirements=self._satisfied,
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

    # ── internal ───────────────────────────────────────────────────────

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
            last_action_result=self._last_action_result,
            step_count=self._step_count,
        )

    def _apply_action(self, action: Action) -> None:
        at = action.action_type
        rationale = action.rationale or at.value

        if at == ActionType.SELECT_PREREQUISITE_SIGN:
            self._coverage.has_prerequisite = True
            sign = action.payload.get("sign", "related-sign")
            self._plan.append(f"Prerequisite sign: {sign}")
            self._last_action_result = f"Selected prerequisite sign '{sign}'."

        elif at == ActionType.SLOW_MOTION_DEMO:
            self._coverage.has_timing_support = True
            self._plan.append("Slow-motion demonstration of target sign.")
            self._last_action_result = "Provided slowed demonstration."

        elif at == ActionType.ADD_LOCATION_CUE:
            self._coverage.has_visual_cue = True
            cue = action.payload.get("cue", "spatial marker")
            self._plan.append(f"Location cue: {cue}")
            self._last_action_result = f"Added location cue: {cue}."

        elif at == ActionType.ADD_MOVEMENT_HINT:
            self._coverage.has_movement_hint = True
            hint = action.payload.get("hint", "directional arrow overlay")
            self._plan.append(f"Movement hint: {hint}")
            self._last_action_result = f"Added movement hint: {hint}."

        elif at == ActionType.CHOOSE_FEEDBACK_STYLE:
            self._coverage.has_feedback_style = True
            style = action.payload.get("style", "visual")
            self._plan.append(f"Feedback style: {style}")
            self._last_action_result = f"Set feedback style to '{style}'."

        elif at == ActionType.GENERATE_MICRO_DRILL:
            self._coverage.has_micro_drill = True
            focus = action.payload.get("focus", "target sign")
            self._plan.append(f"Micro-drill on {focus}.")
            self._last_action_result = f"Generated micro-drill focusing on {focus}."

        elif at == ActionType.QUICK_ASSESSMENT:
            self._coverage.has_assessment = True
            self._plan.append("Quick learner assessment checkpoint.")
            self._last_action_result = "Assessment checkpoint added."

        elif at == ActionType.REVISION_LOOP:
            self._coverage.has_revision = True
            self._plan.append("Revision loop: repeat weak segments.")
            self._last_action_result = "Revision loop scheduled."

        elif at == ActionType.FINALIZE_PLAN:
            self._coverage.plan_finalized = True
            self._plan.append(f"Plan finalized. Rationale: {rationale}")
            self._last_action_result = "Tutoring plan finalized."

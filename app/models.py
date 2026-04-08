from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class ErrorType(str, Enum):
    HANDSHAPE = "handshape"
    MOVEMENT = "movement"
    LOCATION = "location"
    TIMING = "timing"
    ORIENTATION = "orientation"


class ActionType(str, Enum):
    SELECT_PREREQUISITE_SIGN = "select_prerequisite_sign"
    SLOW_MOTION_DEMO = "slow_motion_demo"
    ADD_LOCATION_CUE = "add_location_cue"
    ADD_MOVEMENT_HINT = "add_movement_hint"
    CHOOSE_FEEDBACK_STYLE = "choose_feedback_style"
    GENERATE_MICRO_DRILL = "generate_micro_drill"
    QUICK_ASSESSMENT = "quick_assessment"
    REVISION_LOOP = "revision_loop"
    FINALIZE_PLAN = "finalize_plan"


class FeedbackStyle(str, Enum):
    VISUAL = "visual"
    TACTILE = "tactile"
    VERBAL = "verbal"
    MODELING = "modeling"


class SupportNeed(str, Enum):
    VISUAL_AIDS = "visual_aids"
    SLOWED_PACING = "slowed_pacing"
    REPETITION = "repetition"
    TACTILE_GUIDANCE = "tactile_guidance"
    ATTENTION_SUPPORT = "attention_support"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ── Domain models ──────────────────────────────────────────────────────

class LearnerProfile(BaseModel):
    age_band: str
    proficiency: str
    primary_language: str = "ASL"
    support_needs: list[SupportNeed] = []


class LessonGoal(BaseModel):
    target_sign: str
    description: str


class ErrorPattern(BaseModel):
    error_type: ErrorType
    severity: float = Field(ge=0.0, le=1.0)
    description: str


class TutoringConstraint(BaseModel):
    max_steps: int
    required_outputs: list[str]
    comprehension_target: float = Field(
        default=0.65,
        description="Minimum comprehension the learner must reach per error for a pass.",
    )


# ── Hidden learner simulation state ───────────────────────────────────

class LearnerSimState(BaseModel):
    """Internal learner state that evolves during the episode.
    Comprehension per error is HIDDEN unless the agent assesses."""
    comprehension: dict[str, float] = Field(
        default_factory=dict,
        description="Comprehension per ErrorType value — 0.0 (no understanding) to 1.0 (mastered).",
    )
    attention: float = Field(default=1.0, ge=0.0, le=1.0)
    frustration: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SimulationParams(BaseModel):
    """Per-task parameters that control the learner simulation."""
    initial_comprehension: dict[str, float] = {}
    initial_attention: float = 0.9
    initial_frustration: float = 0.1
    initial_confidence: float = 0.5
    attention_decay_per_step: float = 0.06
    default_seed: int = 42


# ── Learner signals (observable by agent) ──────────────────────────────

class LearnerSignals(BaseModel):
    """Qualitative feedback the agent can observe each step."""
    engagement: str = ""
    emotional_state: str = ""
    intervention_feedback: str = ""
    assessed_comprehension: dict[str, float] | None = Field(
        default=None,
        description="Only populated after a quick_assessment action.",
    )


# ── Coverage flags ─────────────────────────────────────────────────────

class CoverageFlags(BaseModel):
    has_prerequisite: bool = False
    has_visual_cue: bool = False
    has_timing_support: bool = False
    has_movement_hint: bool = False
    has_feedback_style: bool = False
    has_micro_drill: bool = False
    has_assessment: bool = False
    has_revision: bool = False
    plan_finalized: bool = False
    demo_before_drill: bool = False
    cue_before_drill: bool = False
    assessed_before_revision: bool = False


# ── Observation (OpenEnv-compliant) ────────────────────────────────────

class Observation(BaseModel):
    done: bool = False
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Domain-specific
    task_id: str = ""
    difficulty: Difficulty = Difficulty.EASY
    learner: LearnerProfile = Field(default_factory=lambda: LearnerProfile(age_band="", proficiency=""))
    lesson_goal: LessonGoal = Field(default_factory=lambda: LessonGoal(target_sign="", description=""))
    error_patterns: list[ErrorPattern] = []
    support_needs: list[SupportNeed] = []
    current_plan: list[str] = []
    completed_requirements: list[str] = []
    remaining_steps: int = 0
    allowed_actions: list[ActionType] = []
    coverage: CoverageFlags = Field(default_factory=CoverageFlags)
    learner_signals: LearnerSignals = Field(default_factory=LearnerSignals)
    step_count: int = 0


# ── Action (OpenEnv-compliant) ─────────────────────────────────────────

class Action(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    action_type: ActionType
    rationale: str = ""
    payload: dict[str, Any] = {}


# ── State (OpenEnv-compliant) ──────────────────────────────────────────

class State(BaseModel):
    episode_id: str | None = None
    step_count: int = 0
    task_id: str = ""
    max_steps: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    coverage: CoverageFlags = Field(default_factory=CoverageFlags)
    current_plan: list[str] = []
    final_grade: GradeReport | None = None


# ── Reward ─────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    comprehension_gain: float = 0.0
    pedagogical_quality: float = 0.0
    learner_wellbeing: float = 0.0
    efficiency: float = 0.0
    total: float = 0.0


# ── Task spec ──────────────────────────────────────────────────────────

class TaskSpec(BaseModel):
    id: str
    difficulty: Difficulty
    learner: LearnerProfile
    lesson_goal: LessonGoal
    error_patterns: list[ErrorPattern]
    constraints: TutoringConstraint
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    grader_weights: dict[str, float] = {}
    ideal_action_order: list[ActionType] = []


# ── Grade report ───────────────────────────────────────────────────────

class GradeReport(BaseModel):
    total_score: float = Field(ge=0.0, le=1.0)
    sub_scores: RewardBreakdown
    passed: bool
    reasoning: str
    comprehension_before: dict[str, float] = {}
    comprehension_after: dict[str, float] = {}
    missing_requirements: list[str] = []


# ── OpenEnv-compliant API models ───────────────────────────────────────

class ResetRequest(BaseModel):
    model_config = {"extra": "allow"}
    seed: int | None = None
    episode_id: str | None = None
    task_id: str | None = None


class ResetResponse(BaseModel):
    observation: dict[str, Any]
    reward: float | None = None
    done: bool = False


class StepRequest(BaseModel):
    model_config = {"extra": "allow"}
    action: dict[str, Any]
    timeout_s: float | None = None


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float | None = None
    done: bool = False


class SchemaResponse(BaseModel):
    action: dict[str, Any]
    observation: dict[str, Any]
    state: dict[str, Any]


class EnvironmentMetadata(BaseModel):
    name: str
    description: str
    version: str | None = None
    author: str | None = None


class HealthResponse(BaseModel):
    status: str = "healthy"


class TaskSummary(BaseModel):
    id: str
    difficulty: Difficulty
    target_sign: str
    description: str
    max_steps: int


# Fix forward reference
State.model_rebuild()

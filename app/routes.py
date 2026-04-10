from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, HTTPException

from app.env import TutoringEnv
from app.models import (
    Action,
    EnvironmentMetadata,
    HealthResponse,
    Observation,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    State,
    StepRequest,
    StepResponse,
    TaskSummary,
)
from app.tasks import list_tasks

router = APIRouter()

# Thread-safe session management: each episode gets its own TutoringEnv
_sessions: dict[str, TutoringEnv] = {}
_sessions_lock = threading.Lock()
_active_env: TutoringEnv | None = None  # most-recent session shortcut


def _get_env() -> TutoringEnv:
    if _active_env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _active_env


# ── OpenEnv-standard endpoints ─────────────────────────────────────────

@router.get("/")
def root() -> dict:
    return {
        "status": "healthy",
        "name": "SignAdapt",
        "message": "OpenEnv environment is running.",
    }

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.get("/metadata", response_model=EnvironmentMetadata)
def metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="SignAdapt",
        description=(
            "Adaptive sign-language tutoring environment with PyTorch-powered "
            "learner simulation. An agent plans step-by-step teaching "
            "interventions for deaf/hard-of-hearing learners, adapting to "
            "error patterns, learner fatigue, memory decay, and support needs. "
            "Features 6 tasks across 4 difficulty levels with outcome-based grading."
        ),
        version="2.0.0",
        author="SignAdapt Team",
    )


@router.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return SchemaResponse(
        action=Action.model_json_schema(),
        observation=Observation.model_json_schema(),
        state=State.model_json_schema(),
    )


@router.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()) -> ResetResponse:
    global _active_env
    try:
        env = TutoringEnv()
        obs = env.reset(task_id=request.task_id, episode_id=request.episode_id, seed=request.seed)
        with _sessions_lock:
            _sessions[env._episode_id] = env
            _active_env = env
        return ResetResponse(
            observation=obs.model_dump(),
            reward=None,
            done=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    env = _get_env()
    try:
        action_data = request.action
        action = Action(**action_data)
        obs = env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=env.last_step_reward,
            done=env.is_done,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/state")
def state() -> dict:
    env = _get_env()
    return env.state.model_dump()


# ── Extra endpoints ────────────────────────────────────────────────────

@router.get("/tasks", response_model=list[TaskSummary])
def tasks() -> list[TaskSummary]:
    return list_tasks()


@router.get("/analytics")
def analytics() -> dict[str, Any]:
    """Episode analytics — shows action effectiveness and learner trajectory."""
    env = _get_env()
    grade = env.final_grade
    st = env.state
    return {
        "episode_id": st.episode_id,
        "task_id": st.task_id,
        "steps_taken": st.step_count,
        "max_steps": st.max_steps,
        "done": st.done,
        "cumulative_reward": st.cumulative_reward,
        "coverage": st.coverage.model_dump(),
        "final_grade": grade.model_dump() if grade else None,
        "action_history": [a.value for a in env._action_history],
        "sessions_active": len(_sessions),
    }

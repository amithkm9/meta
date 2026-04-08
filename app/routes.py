from __future__ import annotations

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
env = TutoringEnv()


# ── OpenEnv-standard endpoints ─────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.get("/metadata", response_model=EnvironmentMetadata)
def metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="SignAdapt",
        description=(
            "Adaptive sign-language tutoring environment. An agent plans "
            "step-by-step teaching interventions for deaf/hard-of-hearing "
            "learners, adapting to error patterns and support needs."
        ),
        version="1.0.0",
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
    try:
        obs = env.reset(task_id=request.task_id, episode_id=request.episode_id, seed=request.seed)
        return ResetResponse(
            observation=obs.model_dump(),
            reward=None,
            done=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
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
    return env.state.model_dump()


# ── Extra endpoints ────────────────────────────────────────────────────

@router.get("/tasks", response_model=list[TaskSummary])
def tasks() -> list[TaskSummary]:
    return list_tasks()

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.env import TutoringEnv
from app.models import (
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResult,
    TaskSummary,
)
from app.tasks import list_tasks

router = APIRouter()
env = TutoringEnv()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/tasks", response_model=list[TaskSummary])
def tasks() -> list[TaskSummary]:
    return list_tasks()


@router.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    try:
        return env.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    try:
        return env.step(episode_id=req.episode_id, action=req.action)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return env.state()

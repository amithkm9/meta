from __future__ import annotations

import json
from pathlib import Path

from app.models import TaskSpec, TaskSummary

_TASKS_PATH = Path(__file__).parent / "sample_data" / "tasks.json"

_task_cache: list[TaskSpec] | None = None


def _load() -> list[TaskSpec]:
    global _task_cache
    if _task_cache is None:
        raw = json.loads(_TASKS_PATH.read_text())
        _task_cache = [TaskSpec(**t) for t in raw]
    return _task_cache


def list_tasks() -> list[TaskSummary]:
    return [
        TaskSummary(
            id=t.id,
            difficulty=t.difficulty,
            target_sign=t.lesson_goal.target_sign,
            description=t.lesson_goal.description,
            max_steps=t.constraints.max_steps,
        )
        for t in _load()
    ]


def get_task(task_id: str) -> TaskSpec:
    for t in _load():
        if t.id == task_id:
            return t
    raise ValueError(f"Unknown task: {task_id}")


def default_task_id() -> str:
    return _load()[0].id

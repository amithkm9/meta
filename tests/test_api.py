"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_tasks():
    resp = client.get("/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) >= 3
    ids = {t["id"] for t in tasks}
    assert "easy_remediate_handshape" in ids
    assert "medium_fix_movement_with_scaffold" in ids
    assert "hard_adaptive_multi_error_plan" in ids


def test_reset():
    resp = client.post("/reset", json={"task_id": "easy_remediate_handshape"})
    assert resp.status_code == 200
    data = resp.json()
    assert "episode_id" in data
    assert data["done"] is False
    assert data["observation"]["task_id"] == "easy_remediate_handshape"


def test_step():
    reset_resp = client.post("/reset", json={"task_id": "easy_remediate_handshape"}).json()
    eid = reset_resp["episode_id"]

    step_resp = client.post("/step", json={
        "episode_id": eid,
        "action": {
            "action_type": "slow_motion_demo",
            "rationale": "test",
            "payload": {},
        },
    })
    assert step_resp.status_code == 200
    data = step_resp.json()
    assert "reward" in data
    assert "observation" in data
    assert data["done"] is False


def test_state():
    client.post("/reset", json={"task_id": "easy_remediate_handshape"})
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "easy_remediate_handshape"
    assert data["done"] is False


def test_full_episode():
    reset_resp = client.post("/reset", json={"task_id": "easy_remediate_handshape"}).json()
    eid = reset_resp["episode_id"]

    actions = [
        {"action_type": "slow_motion_demo", "rationale": "demo", "payload": {}},
        {"action_type": "add_movement_hint", "rationale": "hint", "payload": {}},
        {"action_type": "choose_feedback_style", "rationale": "fb", "payload": {"style": "visual"}},
        {"action_type": "quick_assessment", "rationale": "check", "payload": {}},
        {"action_type": "finalize_plan", "rationale": "done", "payload": {}},
    ]

    for action in actions:
        resp = client.post("/step", json={"episode_id": eid, "action": action})
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["reward"] <= 1.0
        if data["done"]:
            assert "final_grade" in data["info"]
            assert 0.0 <= data["info"]["final_grade"]["total_score"] <= 1.0
            break


def test_reset_invalid_task():
    resp = client.post("/reset", json={"task_id": "nonexistent_task"})
    assert resp.status_code == 400

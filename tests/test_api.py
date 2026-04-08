"""Tests for the FastAPI endpoints — OpenEnv-compliant."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_metadata():
    resp = client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["name"], str)
    assert isinstance(data["description"], str)


def test_schema():
    resp = client.get("/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "action" in data and isinstance(data["action"], dict)
    assert "observation" in data and isinstance(data["observation"], dict)
    assert "state" in data and isinstance(data["state"], dict)


def test_tasks():
    resp = client.get("/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) >= 3
    ids = {t["id"] for t in tasks}
    assert "easy_remediate_handshape" in ids
    assert "medium_movement_timing_scaffold" in ids
    assert "hard_multi_error_adaptive" in ids


def test_reset():
    resp = client.post("/reset", json={"task_id": "easy_remediate_handshape"})
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert data["done"] is False
    assert data["reward"] is None
    obs = data["observation"]
    assert obs["task_id"] == "easy_remediate_handshape"
    assert "learner_signals" in obs


def test_reset_empty_body():
    resp = client.post("/reset", json={})
    assert resp.status_code == 200
    assert resp.json()["done"] is False


def test_step():
    client.post("/reset", json={"task_id": "easy_remediate_handshape"})
    step_resp = client.post("/step", json={
        "action": {"action_type": "slow_motion_demo", "rationale": "test", "payload": {}},
    })
    assert step_resp.status_code == 200
    data = step_resp.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert data["done"] is False
    # Should have learner signals
    obs = data["observation"]
    assert obs["learner_signals"]["intervention_feedback"] != ""


def test_state():
    client.post("/reset", json={"task_id": "easy_remediate_handshape"})
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "easy_remediate_handshape"
    assert "episode_id" in data
    assert "step_count" in data


def test_full_episode_with_assessment():
    """Run a full episode that uses assessment to reveal learner state."""
    client.post("/reset", json={"task_id": "easy_remediate_handshape"})

    actions = [
        {"action_type": "quick_assessment", "rationale": "check", "payload": {}},
        {"action_type": "slow_motion_demo", "rationale": "demo", "payload": {}},
        {"action_type": "generate_micro_drill", "rationale": "drill", "payload": {}},
        {"action_type": "choose_feedback_style", "rationale": "fb", "payload": {"style": "visual"}},
        {"action_type": "quick_assessment", "rationale": "recheck", "payload": {}},
        {"action_type": "finalize_plan", "rationale": "done", "payload": {}},
    ]

    assessed_once = False
    for action in actions:
        resp = client.post("/step", json={"action": action})
        assert resp.status_code == 200
        data = resp.json()
        reward = data["reward"]
        assert reward is None or 0.0 <= reward <= 1.0

        # Assessment should reveal comprehension
        if action["action_type"] == "quick_assessment":
            obs = data["observation"]
            assessed_comp = obs["learner_signals"]["assessed_comprehension"]
            assert assessed_comp is not None
            assessed_once = True

        if data["done"]:
            obs = data["observation"]
            grade = obs.get("metadata", {}).get("final_grade")
            assert grade is not None
            assert 0.0 <= grade["total_score"] <= 1.0
            assert "comprehension_before" in grade
            assert "comprehension_after" in grade
            break

    assert assessed_once


def test_reset_invalid_task():
    resp = client.post("/reset", json={"task_id": "nonexistent_task"})
    assert resp.status_code == 400


def test_openapi_json():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "info" in data
    assert "version" in data["info"]
    paths = data.get("paths", {})
    assert "/reset" in paths
    assert "/step" in paths
    assert "/state" in paths
    assert "/health" in paths
    assert "/metadata" in paths
    assert "/schema" in paths

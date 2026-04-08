#!/usr/bin/env python3
"""SignAdapt baseline inference agent.

Uses an OpenAI-compatible LLM to step through a tutoring episode.
Falls back to a heuristic policy when the LLM output is invalid.
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MAX_SAFETY_STEPS = 15

VALID_ACTIONS = [
    "select_prerequisite_sign",
    "slow_motion_demo",
    "add_location_cue",
    "add_movement_hint",
    "choose_feedback_style",
    "generate_micro_drill",
    "quick_assessment",
    "revision_loop",
    "finalize_plan",
]

# ── LLM client ─────────────────────────────────────────────────────────

client = OpenAI(
    api_key=OPENAI_API_KEY or HF_TOKEN,
    base_url=API_BASE_URL,
)

# ── Heuristic fallback ─────────────────────────────────────────────────


def heuristic_action(obs: dict) -> dict:
    """Simple rule-based fallback policy."""
    coverage = obs.get("coverage", {})
    difficulty = obs.get("difficulty", "easy")
    error_types = [ep["error_type"] for ep in obs.get("error_patterns", [])]
    remaining = obs.get("remaining_steps", 1)

    if remaining <= 1:
        return {"action_type": "finalize_plan", "rationale": "Last step — finalizing.", "payload": {}}

    if difficulty == "hard" and not coverage.get("has_prerequisite"):
        return {"action_type": "select_prerequisite_sign", "rationale": "Hard task needs prerequisite.", "payload": {"sign": "foundational-sign"}}

    if not coverage.get("has_timing_support"):
        return {"action_type": "slow_motion_demo", "rationale": "Demonstrate target sign slowly.", "payload": {}}

    if not coverage.get("has_visual_cue"):
        return {"action_type": "add_location_cue", "rationale": "Add visual cue for learner.", "payload": {"cue": "body-midline marker"}}

    if not coverage.get("has_movement_hint"):
        return {"action_type": "add_movement_hint", "rationale": "Provide movement direction.", "payload": {"hint": "directional arrow overlay"}}

    if not coverage.get("has_micro_drill"):
        return {"action_type": "generate_micro_drill", "rationale": "Practice drill needed.", "payload": {"focus": "target sign"}}

    if not coverage.get("has_feedback_style"):
        return {"action_type": "choose_feedback_style", "rationale": "Select appropriate feedback.", "payload": {"style": "visual"}}

    if not coverage.get("has_assessment"):
        return {"action_type": "quick_assessment", "rationale": "Check learner progress.", "payload": {}}

    if not coverage.get("has_revision") and difficulty in ("medium", "hard"):
        return {"action_type": "revision_loop", "rationale": "Reinforce weak segments.", "payload": {}}

    return {"action_type": "finalize_plan", "rationale": "All interventions placed.", "payload": {}}


# ── LLM action selection ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a sign-language tutoring planner. Given the current observation, choose exactly ONE next tutoring action.

Valid action_type values: select_prerequisite_sign, slow_motion_demo, add_location_cue, add_movement_hint, choose_feedback_style, generate_micro_drill, quick_assessment, revision_loop, finalize_plan.

Respond with ONLY a JSON object:
{"action_type": "...", "rationale": "...", "payload": {...}}

Rules:
- Pick the most useful next intervention given the learner errors and what is already covered.
- If all required outputs are satisfied, choose finalize_plan.
- If only 1 step remains, choose finalize_plan.
- Keep rationale under 30 words.
"""


def llm_select_action(obs: dict) -> dict | None:
    """Ask the LLM for the next action. Returns None on failure."""
    obs_summary = json.dumps({
        "task_id": obs.get("task_id"),
        "difficulty": obs.get("difficulty"),
        "error_patterns": obs.get("error_patterns"),
        "support_needs": obs.get("support_needs"),
        "coverage": obs.get("coverage"),
        "completed_requirements": obs.get("completed_requirements"),
        "remaining_steps": obs.get("remaining_steps"),
        "current_plan": obs.get("current_plan"),
        "last_action_result": obs.get("last_action_result"),
    }, indent=2)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Observation:\n{obs_summary}"},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        if parsed.get("action_type") in VALID_ACTIONS:
            return {
                "action_type": parsed["action_type"],
                "rationale": str(parsed.get("rationale", ""))[:200],
                "payload": parsed.get("payload", {}),
            }
    except Exception as e:
        print(f"  [WARN] LLM call failed: {e}", file=sys.stderr)
    return None


# ── Main loop ──────────────────────────────────────────────────────────


def run_episode(task_id: str) -> dict:
    http = httpx.Client(base_url=ENV_URL, timeout=30)

    # Reset
    reset_resp = http.post("/reset", json={"task_id": task_id}).json()
    episode_id = reset_resp["episode_id"]
    obs = reset_resp["observation"]
    done = reset_resp["done"]

    print(f"[START] task={task_id} episode={episode_id}")

    step_num = 0
    final_info = {}

    while not done and step_num < MAX_SAFETY_STEPS:
        # Try LLM, fall back to heuristic
        action = llm_select_action(obs)
        if action is None:
            action = heuristic_action(obs)

        print(f"[STEP] step={step_num} action={action['action_type']} rationale={action['rationale']}")

        step_resp = http.post("/step", json={
            "episode_id": episode_id,
            "action": action,
        }).json()

        obs = step_resp["observation"]
        reward = step_resp["reward"]
        done = step_resp["done"]
        info = step_resp.get("info", {})
        final_info = info

        print(f"[STEP] step={step_num} reward={reward} done={done}")
        step_num += 1

    grade = final_info.get("final_grade", {})
    score = grade.get("total_score", 0.0)
    passed = grade.get("passed", False)

    print(f"[END] task={task_id} score={score} passed={passed}")
    return grade


def main() -> None:
    task_ids = [
        "easy_remediate_handshape",
        "medium_fix_movement_with_scaffold",
        "hard_adaptive_multi_error_plan",
    ]
    results = {}
    for tid in task_ids:
        grade = run_episode(tid)
        results[tid] = grade
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tid, g in results.items():
        print(f"  {tid}: score={g.get('total_score', 0.0):.4f}  passed={g.get('passed', False)}")


if __name__ == "__main__":
    main()

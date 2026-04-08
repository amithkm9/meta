#!/usr/bin/env python3
"""SignAdapt baseline inference agent.

Assessment-aware adaptive baseline that reads learner signals,
uses comprehension data from assessments, and adapts its strategy.
Falls back to heuristic when the LLM is unavailable.

Emits structured stdout logs: [START], [STEP], [END].
"""

from __future__ import annotations

import json
import os
import sys

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
    "select_prerequisite_sign", "slow_motion_demo", "add_location_cue",
    "add_movement_hint", "choose_feedback_style", "generate_micro_drill",
    "quick_assessment", "revision_loop", "finalize_plan",
]

client = OpenAI(api_key=OPENAI_API_KEY or HF_TOKEN, base_url=API_BASE_URL)

# ── Heuristic: assessment-aware adaptive policy ────────────────────────


def heuristic_action(obs: dict) -> dict:
    """Smart rule-based policy that uses learner signals and adapts."""
    coverage = obs.get("coverage", {})
    difficulty = obs.get("difficulty", "easy")
    remaining = obs.get("remaining_steps", 1)
    signals = obs.get("learner_signals", {})
    emotional = signals.get("emotional_state", "")
    assessed = signals.get("assessed_comprehension")
    error_types = [ep["error_type"] for ep in obs.get("error_patterns", [])]

    # Always finalize on last step
    if remaining <= 1:
        return _act("finalize_plan", "Last step — finalizing.")

    # STRATEGY 1: Assess first on medium/hard (information before action)
    if not coverage.get("has_assessment") and remaining > 3 and difficulty != "easy":
        return _act("quick_assessment", "Assess learner state before intervening.")

    # STRATEGY 2: If learner is frustrated, prioritize encouragement
    if "frustration" in emotional and not coverage.get("has_feedback_style"):
        return _act("choose_feedback_style", "Learner is frustrated — provide encouragement.", {"style": "visual"})

    # STRATEGY 3: Hard tasks need prerequisite foundation
    if difficulty == "hard" and not coverage.get("has_prerequisite"):
        return _act("select_prerequisite_sign", "Build foundation for complex sign.", {"sign": "foundational-sign"})

    # STRATEGY 4: Demo first (always high value, boosts attention)
    if not coverage.get("has_timing_support"):
        return _act("slow_motion_demo", "Demonstrate sign slowly — engages learner and helps movement/timing.")

    # STRATEGY 5: Target weakest comprehension area if assessed
    if assessed:
        weakest = min(assessed, key=assessed.get)
        weakest_val = assessed[weakest]
        if weakest_val < 0.5:
            if weakest in ("location",) and not coverage.get("has_visual_cue"):
                return _act("add_location_cue", f"Target weak area: {weakest}.", {"cue": "body-midline marker"})
            if weakest in ("movement", "orientation") and not coverage.get("has_movement_hint"):
                return _act("add_movement_hint", f"Target weak area: {weakest}.", {"hint": "directional overlay"})

    # STRATEGY 6: Add cues and hints (visual cues help all learners, not just location errors)
    if not coverage.get("has_visual_cue"):
        return _act("add_location_cue", "Visual cue helps learner ground the sign spatially.", {"cue": "body-midline marker"})

    if not coverage.get("has_movement_hint"):
        return _act("add_movement_hint", "Movement hint reinforces correct form.", {"hint": "directional overlay"})

    # STRATEGY 7: Drill after demo and cues (sequence bonus)
    if not coverage.get("has_micro_drill"):
        return _act("generate_micro_drill", "Practice drill — demo and cues given first.", {"focus": "target sign"})

    # STRATEGY 8: Feedback if not yet given
    if not coverage.get("has_feedback_style"):
        return _act("choose_feedback_style", "Select encouraging feedback style.", {"style": "visual"})

    # STRATEGY 9: Re-assess to check progress, then revise if needed
    if not coverage.get("has_revision") and difficulty in ("medium", "hard"):
        # If we haven't assessed recently, assess first for targeted revision
        if remaining > 2 and coverage.get("has_assessment"):
            return _act("quick_assessment", "Re-assess before revision for targeted approach.")
        return _act("revision_loop", "Reinforce weak areas through repetition.")

    # Done — finalize
    return _act("finalize_plan", "All interventions placed — finalizing plan.")


def _act(action_type: str, rationale: str, payload: dict | None = None) -> dict:
    return {"action_type": action_type, "rationale": rationale, "payload": payload or {}}


# ── LLM action selection ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert sign-language tutoring planner. You adapt your teaching strategy based on learner feedback.

Given the observation (including learner signals like engagement, emotional state, and assessed comprehension), choose exactly ONE next action.

Valid actions: select_prerequisite_sign, slow_motion_demo, add_location_cue, add_movement_hint, choose_feedback_style, generate_micro_drill, quick_assessment, revision_loop, finalize_plan.

Key strategies:
- ASSESS FIRST to understand learner state before teaching blindly
- If learner is frustrated, use choose_feedback_style to encourage
- Demo BEFORE drill for better learning (sequence bonus)
- Assess BEFORE revision for targeted practice (sequence bonus)
- Target the weakest comprehension area revealed by assessment
- Manage attention: don't overload with drills, use demos and feedback to re-engage
- Finalize when remaining_steps <= 1 or all interventions are in place

Respond with ONLY a JSON object: {"action_type": "...", "rationale": "...", "payload": {...}}
Keep rationale under 30 words."""


def llm_select_action(obs: dict) -> dict | None:
    obs_summary = json.dumps({
        "task_id": obs.get("task_id"),
        "difficulty": obs.get("difficulty"),
        "error_patterns": obs.get("error_patterns"),
        "support_needs": obs.get("support_needs"),
        "coverage": obs.get("coverage"),
        "completed_requirements": obs.get("completed_requirements"),
        "remaining_steps": obs.get("remaining_steps"),
        "current_plan": obs.get("current_plan"),
        "learner_signals": obs.get("learner_signals"),
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

    reset_resp = http.post("/reset", json={"task_id": task_id}).json()
    obs = reset_resp["observation"]
    done = reset_resp.get("done", False)
    episode_id = obs.get("metadata", {}).get("episode_id", "unknown")

    print(f"[START] task={task_id} episode={episode_id}")

    step_num = 0
    final_grade = {}

    while not done and step_num < MAX_SAFETY_STEPS:
        action = llm_select_action(obs)
        if action is None:
            action = heuristic_action(obs)

        print(f"[STEP] step={step_num} action={action['action_type']} rationale={action['rationale']}")

        step_resp = http.post("/step", json={"action": action}).json()
        obs = step_resp["observation"]
        reward = step_resp.get("reward", 0.0) or 0.0
        done = step_resp.get("done", False)

        grade_data = obs.get("metadata", {}).get("final_grade")
        if grade_data:
            final_grade = grade_data

        print(f"[STEP] step={step_num} reward={reward} done={done}")
        step_num += 1

    score = final_grade.get("total_score", 0.0) if final_grade else 0.0
    passed = final_grade.get("passed", False) if final_grade else False

    print(f"[END] task={task_id} score={score} passed={passed}")
    return final_grade


def main() -> None:
    task_ids = [
        "easy_remediate_handshape",
        "medium_movement_timing_scaffold",
        "hard_multi_error_adaptive",
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
        score = g.get("total_score", 0.0) if g else 0.0
        passed = g.get("passed", False) if g else False
        comp_b = g.get("comprehension_before", {}) if g else {}
        comp_a = g.get("comprehension_after", {}) if g else {}
        print(f"  {tid}:")
        print(f"    score={score:.4f}  passed={passed}")
        print(f"    comprehension: {comp_b} → {comp_a}")


if __name__ == "__main__":
    main()

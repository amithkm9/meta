#!/usr/bin/env python3
"""
SignAdapt Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"

ENV_URL = os.getenv("ENV_URL") or "http://localhost:7860"
BENCHMARK = "signadapt"
MAX_SAFETY_STEPS = 15

VALID_ACTIONS = [
    "select_prerequisite_sign", "slow_motion_demo", "add_location_cue",
    "add_movement_hint", "choose_feedback_style", "generate_micro_drill",
    "quick_assessment", "revision_loop", "finalize_plan",
]

TASK_IDS = [
    "easy_remediate_handshape",
    "medium_movement_timing_scaffold",
    "hard_multi_error_adaptive",
]

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Structured logging (exact OpenEnv format) ────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── Heuristic: assessment-aware adaptive policy ──────────────────────


def heuristic_action(obs: dict) -> dict:
    """Smart rule-based policy that uses learner signals and adapts."""
    coverage = obs.get("coverage", {})
    difficulty = obs.get("difficulty", "easy")
    remaining = obs.get("remaining_steps", 1)
    signals = obs.get("learner_signals", {})
    emotional = signals.get("emotional_state", "")
    assessed = signals.get("assessed_comprehension")

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

    # STRATEGY 6: Add cues and hints
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
        if remaining > 2 and coverage.get("has_assessment"):
            return _act("quick_assessment", "Re-assess before revision for targeted approach.")
        return _act("revision_loop", "Reinforce weak areas through repetition.")

    # Done — finalize
    return _act("finalize_plan", "All interventions placed — finalizing plan.")


def _act(action_type: str, rationale: str, payload: dict | None = None) -> dict:
    return {"action_type": action_type, "rationale": rationale, "payload": payload or {}}


# ── LLM action selection ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert sign-language tutoring planner. You plan adaptive interventions for a simulated deaf/hard-of-hearing learner.

The learner has hidden comprehension state. You see engagement signals and can reveal comprehension via quick_assessment.

VALID ACTIONS (choose exactly ONE per step):
- select_prerequisite_sign: Build foundational understanding (1.25x boost on future actions). Best for hard tasks.
- slow_motion_demo: Demonstrate the sign slowly. Strong for movement/timing errors. Boosts attention. Do BEFORE drill for 1.3x bonus.
- add_location_cue: Visual spatial cue. Strong for location errors. Payload: {"cue": "description"}.
- add_movement_hint: Movement/orientation guidance. Payload: {"hint": "description"}.
- choose_feedback_style: Encouragement — reduces frustration, boosts attention. Payload: {"style": "visual"}.
- generate_micro_drill: Practice drill. Most effective AFTER demo/cues (sequence bonus). Payload: {"focus": "target sign"}.
- quick_assessment: Reveals actual comprehension scores per error type. Slight frustration increase. Do BEFORE revision for 1.25x bonus.
- revision_loop: Targets weakest comprehension area. Best after assessment.
- finalize_plan: End the episode and trigger grading. Use when remaining_steps <= 1 or all interventions done.

STRATEGY GUIDE:
1. On medium/hard tasks: assess first to see which errors are weakest
2. If learner_signals.emotional_state mentions frustration: prioritize choose_feedback_style
3. Sequence matters: demo -> drill (1.3x), assess -> revision (1.25x), prerequisite -> all (1.25x)
4. Don't repeat the same action — diminishing returns and efficiency penalty
5. Check remaining_steps — finalize on the last step or you lose the episode
6. Check coverage to see what has already been done — don't repeat covered interventions

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


# ── Main loop ────────────────────────────────────────────────────────

def run_episode(task_id: str) -> dict:
    http = httpx.Client(base_url=ENV_URL, timeout=30)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_grade = {}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = http.post("/reset", json={"task_id": task_id}).json()
        obs = reset_resp["observation"]
        done = reset_resp.get("done", False)

        for step in range(1, MAX_SAFETY_STEPS + 1):
            if done:
                break

            action = llm_select_action(obs)
            if action is None:
                action = heuristic_action(obs)

            error_msg = None
            try:
                step_resp = http.post("/step", json={"action": action}).json()
                obs = step_resp["observation"]
                reward = step_resp.get("reward", 0.0) or 0.0
                done = step_resp.get("done", False)

                grade_data = obs.get("metadata", {}).get("final_grade")
                if grade_data:
                    final_grade = grade_data
            except Exception as e:
                reward = 0.0
                done = False
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action["action_type"], reward=reward, done=done, error=error_msg)

            if done:
                break

        score = final_grade.get("total_score", 0.0) if final_grade else 0.0
        success = final_grade.get("passed", False) if final_grade else False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return final_grade


def main() -> None:
    results = {}
    for tid in TASK_IDS:
        grade = run_episode(tid)
        results[tid] = grade
        print(flush=True)

    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    total = 0.0
    for tid, g in results.items():
        s = g.get("total_score", 0.0) if g else 0.0
        p = g.get("passed", False) if g else False
        total += s
        print(f"  {tid}: score={s:.2f} passed={str(p).lower()}", flush=True)
    avg = total / len(results) if results else 0.0
    print(f"  average_score={avg:.2f}", flush=True)


if __name__ == "__main__":
    main()

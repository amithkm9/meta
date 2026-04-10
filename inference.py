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
import time
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# The judges provide HF_TOKEN as the API key for their LiteLLM proxy
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "no-key"

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "signadapt"
MAX_SAFETY_STEPS = 15

print(f"  [INFO] Using API_BASE_URL={API_BASE_URL}", file=sys.stderr)
print(f"  [INFO] Using MODEL_NAME={MODEL_NAME}", file=sys.stderr)
print(f"  [INFO] API_KEY={'set (' + API_KEY[:8] + '...)' if API_KEY != 'no-key' else 'NOT SET'}", file=sys.stderr)
print(f"  [INFO] ENV_URL={ENV_URL}", file=sys.stderr)

VALID_ACTIONS = [
    "select_prerequisite_sign", "slow_motion_demo", "add_location_cue",
    "add_movement_hint", "choose_feedback_style", "generate_micro_drill",
    "quick_assessment", "revision_loop", "finalize_plan",
]

TASK_IDS = [
    "easy_remediate_handshape",
    "medium_movement_timing_scaffold",
    "medium_orientation_spatial",
    "hard_multi_error_adaptive",
    "hard_kinesthetic_learner",
    "expert_sentence_flow_fatigue",
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


# ── Adaptive system prompts per difficulty ─────────────────────────────

_DIFFICULTY_STRATEGIES = {
    "easy": """EASY TASK STRATEGY:
- The learner has only 1 error and high attention. Budget is tight (6 steps).
- Skip assessment — the learner is engaged and you only need to fix one thing.
- Go straight to demo + cue/hint, then drill, then finalize.
- Prioritize efficiency: fewer unique actions that all contribute to the single error.""",

    "medium": """MEDIUM TASK STRATEGY:
- 2 error types to fix with 8 steps. Learner has moderate attention.
- ASSESS FIRST to see which error is worse, then target it.
- Sequence: assess -> demo -> targeted cues/hints -> drill -> feedback -> finalize.
- Demo before drill gives 1.3x bonus. Assess before revision gives 1.25x.
- Watch frustration signals — add feedback if emotional_state mentions frustration.""",

    "hard": """HARD TASK STRATEGY:
- 3 error types, 10 steps, learner has LOW attention and HIGH frustration.
- ASSESS FIRST, then immediately address frustration with feedback.
- Build foundation with prerequisite (1.25x boost on ALL future actions).
- Then: demo -> location cue -> movement hint -> drill -> re-assess -> revision -> finalize.
- The learner tires quickly — prioritize high-impact actions early.""",

    "expert": """EXPERT TASK STRATEGY:
- 4 error types, 12 steps, learner has very low attention, high frustration, and FATIGUE.
- CRITICAL: fatigue reduces ALL intervention effectiveness over time. Act fast.
- Memory decay means comprehension fades each step — reinforce with drill/revision.
- ASSESS FIRST to prioritize. ADDRESS FRUSTRATION IMMEDIATELY (feedback early).
- Then: prerequisite -> demo -> location cue -> movement hint -> drill.
- Re-assess mid-episode to track decay. Revision loop for weakest area.
- Balance emotional support with instruction — a burned-out learner learns nothing.""",
}

SYSTEM_PROMPT_BASE = """You are an expert sign-language tutoring planner. You plan adaptive interventions for a simulated deaf/hard-of-hearing learner.

The learner has hidden comprehension state. You see engagement signals and can reveal comprehension via quick_assessment.

VALID ACTIONS (choose exactly ONE per step):
- select_prerequisite_sign: Build foundational understanding (1.25x boost on future actions). Payload: {"sign": "name"}.
- slow_motion_demo: Demonstrate the sign slowly. Strong for movement/timing errors. Boosts attention. Do BEFORE drill for 1.3x bonus.
- add_location_cue: Visual spatial cue. Strong for location errors. Payload: {"cue": "description"}.
- add_movement_hint: Movement/orientation guidance. Payload: {"hint": "description"}.
- choose_feedback_style: Encouragement — reduces frustration, boosts attention. Payload: {"style": "visual"}.
- generate_micro_drill: Practice drill. Most effective AFTER demo/cues (sequence bonus). Payload: {"focus": "target sign"}.
- quick_assessment: Reveals actual comprehension scores per error type. Slight frustration increase. Do BEFORE revision for 1.25x bonus.
- revision_loop: Targets weakest comprehension area. Best after assessment.
- finalize_plan: End the episode and trigger grading. Use when remaining_steps <= 1 or all interventions done.

CRITICAL RULES:
1. ALWAYS finalize on the last step (remaining_steps <= 1) — missing this loses the entire episode
2. NEVER repeat the same action more than once (efficiency penalty + frustration)
3. Check coverage to see what has already been done — don't repeat covered interventions
4. Sequence matters: demo -> drill (1.3x), assess -> revision (1.25x), prerequisite -> all (1.25x)
5. If learner_signals.emotional_state mentions frustration: prioritize choose_feedback_style
6. If assessed_comprehension is available, target the error type with lowest score

{difficulty_strategy}

Respond with ONLY a JSON object: {"action_type": "...", "rationale": "...", "payload": {...}}
Keep rationale under 30 words."""


def _get_system_prompt(difficulty: str) -> str:
    strategy = _DIFFICULTY_STRATEGIES.get(difficulty, _DIFFICULTY_STRATEGIES["medium"])
    return SYSTEM_PROMPT_BASE.format(difficulty_strategy=strategy)


# ── Heuristic: assessment-aware adaptive policy ──────────────────────


def heuristic_action(obs: dict) -> dict:
    """Smart rule-based policy that uses learner signals and adapts."""
    coverage = obs.get("coverage", {})
    difficulty = obs.get("difficulty", "easy")
    remaining = obs.get("remaining_steps", 1)
    signals = obs.get("learner_signals", {})
    emotional = signals.get("emotional_state", "")
    assessed = signals.get("assessed_comprehension")
    error_patterns = obs.get("error_patterns", [])

    # Always finalize on last step
    if remaining <= 1:
        return _act("finalize_plan", "Last step — finalizing.")

    # STRATEGY 1: Assess first on medium/hard/expert (information before action)
    if not coverage.get("has_assessment") and remaining > 3 and difficulty != "easy":
        return _act("quick_assessment", "Assess learner state before intervening.")

    # STRATEGY 2: If learner is frustrated, prioritize encouragement
    if "frustration" in emotional.lower() and not coverage.get("has_feedback_style"):
        return _act("choose_feedback_style", "Learner is frustrated — provide encouragement.", {"style": "visual"})

    # STRATEGY 3: Hard/expert tasks need prerequisite foundation
    if difficulty in ("hard", "expert") and not coverage.get("has_prerequisite"):
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

    # STRATEGY 6: Add cues and hints based on error patterns
    error_types = [ep.get("error_type", "") if isinstance(ep, dict) else getattr(ep, "error_type", "") for ep in error_patterns]
    if any(et in ("location",) for et in error_types) and not coverage.get("has_visual_cue"):
        return _act("add_location_cue", "Visual cue helps learner ground the sign spatially.", {"cue": "body-midline marker"})

    if any(et in ("movement", "orientation") for et in error_types) and not coverage.get("has_movement_hint"):
        return _act("add_movement_hint", "Movement hint reinforces correct form.", {"hint": "directional overlay"})

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
    if not coverage.get("has_revision") and difficulty in ("medium", "hard", "expert"):
        if remaining > 2 and coverage.get("has_assessment"):
            return _act("quick_assessment", "Re-assess before revision for targeted approach.")
        return _act("revision_loop", "Reinforce weak areas through repetition.")

    # STRATEGY 10: Expert tasks — second revision if budget allows
    if difficulty == "expert" and remaining > 2:
        if not coverage.get("has_feedback_style"):
            return _act("choose_feedback_style", "Final encouragement before closing.", {"style": "visual"})
        return _act("revision_loop", "Extra revision to combat memory decay.")

    # Done — finalize
    return _act("finalize_plan", "All interventions placed — finalizing plan.")


def _act(action_type: str, rationale: str, payload: dict | None = None) -> dict:
    return {"action_type": action_type, "rationale": rationale, "payload": payload or {}}


def _apply_action_guardrails(obs: dict, action: dict, history: list[str]) -> dict | None:
    """Return a corrected action when LLM output is likely low-value; otherwise None."""
    action_type = action.get("action_type")
    if action_type not in VALID_ACTIONS:
        return None

    coverage = obs.get("coverage", {})
    remaining = obs.get("remaining_steps", 1)

    # Last step should always finalize.
    if remaining <= 1 and action_type != "finalize_plan":
        return _act("finalize_plan", "Last step guardrail: finalize episode.")

    # Prevent repeated same-action loops that hurt efficiency and pedagogy.
    if len(history) >= 2 and history[-1] == action_type and history[-2] == action_type:
        return heuristic_action(obs)

    # Avoid repeating already-covered setup actions.
    coverage_map = {
        "slow_motion_demo": "has_timing_support",
        "add_location_cue": "has_visual_cue",
        "add_movement_hint": "has_movement_hint",
        "choose_feedback_style": "has_feedback_style",
        "generate_micro_drill": "has_micro_drill",
        "select_prerequisite_sign": "has_prerequisite",
    }
    flag = coverage_map.get(action_type)
    if flag and coverage.get(flag):
        return heuristic_action(obs)

    return None


# ── LLM action selection with multi-turn context ───────────────────────


def llm_select_action(
    obs: dict,
    difficulty: str,
    conversation_history: list[dict],
    retry_count: int = 0,
) -> dict | None:
    """Select action using LLM with full conversation context.

    Returns None on ANY failure so the caller can fall back to heuristic.
    """
    try:
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
        }, default=str, indent=2)

        messages = [
            {"role": "system", "content": _get_system_prompt(difficulty)},
        ]

        # Add conversation history (last 4 turns max to stay within context)
        for turn in conversation_history[-4:]:
            messages.append(turn)

        messages.append({"role": "user", "content": f"Current observation:\n{obs_summary}"})

        # Single attempt — no retries to avoid timeout on judges' infra
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
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
            else:
                print(f"  [WARN] LLM returned invalid action: {parsed.get('action_type')}", file=sys.stderr)
        except Exception as e:
            print(f"  [WARN] LLM call failed: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  [WARN] LLM selection failed: {e}", file=sys.stderr)
    return None


# ── Main loop ────────────────────────────────────────────────────────

def run_episode(task_id: str) -> dict:
    http = httpx.Client(base_url=ENV_URL, timeout=30)

    rewards: List[float] = []
    action_history: List[str] = []
    conversation_history: list[dict] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_grade = {}
    difficulty = "medium"

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = http.post("/reset", json={"task_id": task_id}).json()
        obs = reset_resp["observation"]
        done = reset_resp.get("done", False)
        difficulty = obs.get("difficulty", "medium") if isinstance(obs, dict) else "medium"

        for step in range(1, MAX_SAFETY_STEPS + 1):
            if done:
                break

            # Select action: try LLM first, fall back to heuristic
            try:
                action = llm_select_action(obs, difficulty, conversation_history)
            except Exception:
                action = None
            if action is None:
                action = heuristic_action(obs)
            else:
                corrected = _apply_action_guardrails(obs, action, action_history)
                if corrected is not None:
                    action = corrected

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
            action_history.append(action["action_type"])
            steps_taken = step

            # Build conversation history for multi-turn context
            try:
                conversation_history.append({
                    "role": "assistant",
                    "content": json.dumps(action, default=str),
                })
                signals = obs.get("learner_signals", {}) if isinstance(obs, dict) else {}
                conversation_history.append({
                    "role": "user",
                    "content": f"Step {step} result: reward={reward:.2f}, engagement={signals.get('engagement', 'unknown')}, emotional={signals.get('emotional_state', 'unknown')}, remaining={obs.get('remaining_steps', 0)}",
                })
            except Exception:
                pass  # Non-critical — conversation history is optional context

            log_step(step=step, action=action["action_type"], reward=reward, done=done, error=error_msg)

            if done:
                break

        score = final_grade.get("total_score", 0.0) if final_grade else 0.0
        success = final_grade.get("passed", False) if final_grade else False

    except Exception as e:
        print(f"  [ERROR] Episode failed: {e}", file=sys.stderr)

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

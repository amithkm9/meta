# SignAdapt — Adaptive Sign-Language Tutoring Environment

An **OpenEnv-compliant** reinforcement learning environment where an AI agent must plan **adaptive sign-language teaching interventions** for a simulated deaf/hard-of-hearing learner whose internal state (comprehension, attention, frustration, confidence) evolves dynamically in response to each intervention.

## What Makes This Different

Most hackathon environments are checklists: do X, get points. **SignAdapt is a genuine simulation.**

The learner has **hidden internal state** that the agent cannot see directly:
- **Comprehension** per error type (0.0–1.0) — how well the learner understands each aspect of the sign
- **Attention** — decays over time, recovers with engaging interventions
- **Frustration** — builds with failed or repetitive interventions, reduced by encouragement
- **Confidence** — grows with successful learning, drops when confused

The agent must:
1. **Assess before acting** — comprehension is hidden unless the agent uses `quick_assessment`
2. **Sequence strategically** — demo before drill gives 1.3x effectiveness; assess before revision gives 1.25x
3. **Manage learner wellbeing** — a frustrated learner learns 40% slower; low attention reduces all gains
4. **Adapt to signals** — qualitative feedback reveals engagement and emotional state without exact numbers
5. **Balance exploration vs exploitation** — spend steps assessing, or trust signals and teach?

**The same action sequence produces different outcomes** depending on learner state, sequencing, and support need alignment. A hardcoded policy cannot achieve maximum score.

## Grounded in ASL Pedagogy

- **Error types** (handshape, movement, location, timing, orientation) reflect real sign production challenges
- **Intervention effectiveness** varies by error type — a slow-motion demo strongly helps movement/timing but barely helps handshape
- **Support need matching** — a visual learner benefits more from location cues; a learner needing repetition benefits from revision loops
- **Sequence bonuses** model real pedagogical best practices: demonstrate before drilling, assess before revising

## Observation Space

Observations follow the OpenEnv standard (`done`, `reward`, `metadata`) plus:

| Field | Description |
|---|---|
| `learner_signals.engagement` | Qualitative attention level (attentive / distracted / disengaged) |
| `learner_signals.emotional_state` | Frustration / confidence signals |
| `learner_signals.intervention_feedback` | How the learner responded to the last action |
| `learner_signals.assessed_comprehension` | **Only revealed after `quick_assessment`** — actual comprehension per error |
| `error_patterns` | What the learner is getting wrong (type, severity) |
| `coverage` | Which intervention types have been applied |
| `remaining_steps` | Budget remaining |

## Action Space

| Action | Effect | Side Effects |
|---|---|---|
| `select_prerequisite_sign` | +0.06 all errors, 1.25x boost on future actions | +attention |
| `slow_motion_demo` | Strong movement/timing gain | +attention, enables drill bonus |
| `add_location_cue` | Strong location gain | +attention for visual learners |
| `add_movement_hint` | Strong movement/orientation gain | neutral |
| `choose_feedback_style` | Small gain all errors | −frustration, +attention |
| `generate_micro_drill` | Good gain if demo/cues given first | −attention (tiring) |
| `quick_assessment` | **Reveals hidden comprehension** | +frustration (mild stress) |
| `revision_loop` | Targets weakest error (1.25x if assessed first) | −attention (repetitive) |
| `finalize_plan` | Triggers final grading | — |

## Tasks

| Task | Difficulty | Errors | Max Steps | Challenge |
|---|---|---|---|---|
| `easy_remediate_handshape` | Easy | 1 error | 6 | Tight budget, simple learner |
| `medium_movement_timing_scaffold` | Medium | 2 errors | 8 | Must assess and target both errors |
| `hard_multi_error_adaptive` | Hard | 3 errors | 10 | Low attention, high frustration, must adapt |

## Grading (Outcome-Based)

The final grade measures **actual learner improvement**, not just plan structure:

| Component | Weight | What It Measures |
|---|---|---|
| Comprehension gain | 0.35–0.40 | Did the learner actually improve? (weighted by severity) |
| Pedagogical quality | 0.20–0.25 | Good sequencing + requirement coverage + pattern bonuses |
| Learner wellbeing | 0.15–0.25 | Attention, frustration, confidence at episode end |
| Efficiency | 0.20 | Fewer steps + no redundant actions |

Pass threshold: **0.70**. The grade includes before/after comprehension scores showing exactly what changed.

## API Endpoints (OpenEnv-Compliant)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status": "healthy"}` |
| GET | `/metadata` | Environment name + description |
| GET | `/schema` | Action/observation/state JSON schemas |
| POST | `/reset` | Start episode: `{"task_id": "...", "seed": 42}` |
| POST | `/step` | Submit action: `{"action": {...}}` |
| GET | `/state` | Episode state (episode_id, step_count, etc.) |
| GET | `/tasks` | List available tasks |

## Inference Script (OpenEnv-Compliant)

The `inference.py` emits structured stdout in the required format:

```
[START] task=<task_name> env=signadapt model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

Uses OpenAI Client for all LLM calls with heuristic fallback when LLM is unavailable.

### Required Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / API key |

## Setup & Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
pytest tests/ -v  # 31 tests

# Inference (server must be running):
export API_BASE_URL=... MODEL_NAME=... HF_TOKEN=...
python inference.py
```

## Docker / HF Spaces

```bash
docker build -t signadapt .
docker run -p 7860:7860 signadapt
```

Binds to `0.0.0.0:7860` — ready for Hugging Face Spaces Docker deployment.

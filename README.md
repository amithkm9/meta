# SignAdapt — Adaptive Sign-Language Tutoring Environment

An **OpenEnv-compliant** reinforcement learning environment where an AI agent learns to plan **adaptive sign-language teaching interventions** for deaf and hard-of-hearing learners.

## Problem Statement

Teaching sign language effectively requires adaptive, sequential decision-making. A tutor must assess learner errors (handshape, movement, location, timing), select appropriate interventions, and build a coherent lesson plan — all while respecting the learner's support needs and cognitive constraints.

**SignAdapt** turns this pedagogical planning problem into a structured environment that an RL or LLM agent can interact with step by step via the standard OpenEnv `reset()` / `step()` / `state()` API.

## Why This Environment Is Real-World

- **Grounded in ASL pedagogy**: Error types (handshape, movement, location, timing, orientation) reflect actual sign production challenges documented in deaf education research.
- **Accessibility-focused**: Learner profiles include support needs like visual aids, slowed pacing, attention support, and tactile guidance.
- **Sequential decision-making**: Good tutoring is adaptive and ordered — not a one-shot prompt. The agent must reason about what the learner needs *next*.
- **Deterministic grading**: All scoring is based on structured coverage flags and requirement tracking, not subjective evaluation. Scores are always in `[0.0, 1.0]`.

## Observation Space

Each observation follows the OpenEnv `Observation` contract (`done`, `reward`, `metadata`) plus domain-specific fields:

| Field | Type | Description |
|---|---|---|
| `done` | bool | Whether the episode has terminated |
| `reward` | float/null | Reward from the last action |
| `metadata` | object | Episode ID, final grade (when done) |
| `task_id` | string | Current task identifier |
| `difficulty` | enum | easy, medium, hard |
| `learner` | object | Age band, proficiency, support needs |
| `lesson_goal` | object | Target sign and description |
| `error_patterns` | list | Error type, severity, description |
| `support_needs` | list | Learner accommodations needed |
| `current_plan` | list | Interventions added so far |
| `completed_requirements` | list | Satisfied grading requirements |
| `remaining_steps` | int | Steps left in budget |
| `coverage` | object | Boolean flags for coverage tracking |
| `allowed_actions` | list | Valid action types |
| `last_action_result` | string | Feedback from previous action |

## Action Space

Actions follow the OpenEnv `Action` contract (`metadata`) plus domain fields:

| Action Type | Purpose |
|---|---|
| `select_prerequisite_sign` | Teach a foundational sign first |
| `slow_motion_demo` | Slowed demonstration for timing/movement |
| `add_location_cue` | Visual spatial marker for location errors |
| `add_movement_hint` | Directional guidance for movement errors |
| `choose_feedback_style` | Select visual/tactile/verbal/modeling feedback |
| `generate_micro_drill` | Focused practice drill |
| `quick_assessment` | Checkpoint learner understanding |
| `revision_loop` | Repeat and reinforce weak segments |
| `finalize_plan` | Submit the tutoring plan for grading |

## Tasks

| Task ID | Difficulty | Target Sign | Max Steps |
|---|---|---|---|
| `easy_remediate_handshape` | Easy | HELLO | 6 |
| `medium_fix_movement_with_scaffold` | Medium | THANK-YOU | 8 |
| `hard_adaptive_multi_error_plan` | Hard | HELP | 10 |

## Reward & Grading

**Step reward** (0.0–1.0) computed per action with weighted components:
- Intervention relevance (0.25) — does the action target active errors?
- Pedagogical sequence (0.25) — is the action in a sensible order?
- Learner need alignment (0.20) — does it match support needs?
- Task completeness (0.20) — fraction of requirements covered
- Efficiency (0.10) — penalizes redundant actions

**Final grade** (0.0–1.0) uses the same components computed over the full episode. Pass threshold: **0.70**.

## API Endpoints (OpenEnv-Compliant)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "healthy"}` |
| GET | `/metadata` | Environment name, description, version |
| GET | `/schema` | JSON schemas for action, observation, state |
| POST | `/reset` | Reset environment, returns initial observation |
| POST | `/step` | Submit action `{"action": {...}}`, returns observation |
| GET | `/state` | Current episode state (episode_id, step_count, etc.) |
| GET | `/tasks` | List available tasks (extra endpoint) |

### Reset Request/Response

```json
// POST /reset
{"task_id": "easy_remediate_handshape"}

// Response
{"observation": {...}, "reward": null, "done": false}
```

### Step Request/Response

```json
// POST /step
{"action": {"action_type": "slow_motion_demo", "rationale": "...", "payload": {}}}

// Response
{"observation": {...}, "reward": 0.45, "done": false}
```

## Local Setup

```bash
cd signadapt
pip install -r requirements.txt
```

## Run the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Run Tests

```bash
pytest tests/ -v
```

## Run Inference

```bash
# Start the server first, then in another terminal:
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python inference.py
```

## Docker

```bash
docker build -t signadapt .
docker run -p 7860:7860 signadapt
```

## Hugging Face Spaces

Deploy as a Docker Space. The app binds to `0.0.0.0:7860` by default, which is the expected port for HF Spaces.

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint (for inference.py) |
| `MODEL_NAME` | Model identifier for inference |
| `HF_TOKEN` | Hugging Face API key |
| `OPENAI_API_KEY` | OpenAI API key (for inference.py) |
| `APP_PORT` | Server port (default: 7860) |

"""PyTorch-based learner behavior model.

A small neural network that predicts comprehension gains given the current
learner state, action, and context. This augments the hand-crafted simulation
with a learned component, making the environment harder to game with a fixed
policy and more realistic in modeling individual learner differences.

The model is pre-trained on synthetic episodes from the hand-crafted simulator
and fine-tuned to capture non-linear interaction effects (e.g., fatigue x
attention x learning style) that are difficult to hand-code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Feature encoding constants ───────────────────────────────────────────

ACTION_TYPES = [
    "select_prerequisite_sign", "slow_motion_demo", "add_location_cue",
    "add_movement_hint", "choose_feedback_style", "generate_micro_drill",
    "quick_assessment", "revision_loop", "finalize_plan",
]

ERROR_TYPES = ["handshape", "movement", "location", "timing", "orientation"]

LEARNING_STYLES = ["visual", "kinesthetic", "auditory", "mixed"]

# Feature dimensions
ACTION_DIM = len(ACTION_TYPES)         # 9  (one-hot)
ERROR_DIM = len(ERROR_TYPES)           # 5  (comprehension per type)
LEARNER_STATE_DIM = 4                  # attention, frustration, confidence, fatigue
CONTEXT_DIM = 6                        # step_frac, has_assessed, has_demo, has_drill, has_feedback, has_prerequisite
STYLE_DIM = len(LEARNING_STYLES)       # 4  (one-hot)
TOTAL_INPUT_DIM = ACTION_DIM + ERROR_DIM + LEARNER_STATE_DIM + CONTEXT_DIM + STYLE_DIM  # 28


class LearnerBehaviorNet(nn.Module):
    """Predicts per-error-type comprehension gains and attention/frustration deltas.

    Input:  28-dim feature vector
    Output: 5 (comp gains per error type) + 2 (attention delta, frustration delta) = 7
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOTAL_INPUT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, ERROR_DIM + 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        # Comprehension gains are sigmoid-scaled to [0, 0.3]
        comp_gains = torch.sigmoid(raw[..., :ERROR_DIM]) * 0.3
        # Attention/frustration deltas are tanh-scaled to [-0.15, 0.15]
        deltas = torch.tanh(raw[..., ERROR_DIM:]) * 0.15
        return torch.cat([comp_gains, deltas], dim=-1)


def encode_state(
    action_type: str,
    comprehension: dict[str, float],
    attention: float,
    frustration: float,
    confidence: float,
    fatigue: float,
    step: int,
    max_steps: int,
    has_assessed: bool,
    has_demo: bool,
    has_drill: bool,
    has_feedback: bool,
    has_prerequisite: bool,
    learning_style: str = "mixed",
) -> torch.Tensor:
    """Encode the current state into a feature vector for the neural model."""
    # Action one-hot
    action_vec = [0.0] * ACTION_DIM
    if action_type in ACTION_TYPES:
        action_vec[ACTION_TYPES.index(action_type)] = 1.0

    # Comprehension per error type
    comp_vec = [comprehension.get(et, 0.0) for et in ERROR_TYPES]

    # Learner state
    learner_vec = [attention, frustration, confidence, fatigue]

    # Context
    step_frac = step / max(max_steps, 1)
    context_vec = [
        step_frac,
        float(has_assessed),
        float(has_demo),
        float(has_drill),
        float(has_feedback),
        float(has_prerequisite),
    ]

    # Learning style one-hot
    style_vec = [0.0] * STYLE_DIM
    if learning_style in LEARNING_STYLES:
        style_vec[LEARNING_STYLES.index(learning_style)] = 1.0
    else:
        style_vec[LEARNING_STYLES.index("mixed")] = 1.0

    features = action_vec + comp_vec + learner_vec + context_vec + style_vec
    return torch.tensor(features, dtype=torch.float32)


# ── Pre-training on synthetic data ────────────────────────────────────────

def _generate_synthetic_dataset(n_episodes: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data by running the hand-crafted simulator.

    Each sample: (state_features) -> (comp_gains_per_error, attention_delta, frustration_delta)
    """
    import random as _random
    from app.env import TutoringEnv, _BASE_EFFECT, _NO_EFFECT_ACTIONS, _LEARNING_STYLE_MODS
    from app.models import ActionType, LearningStyle
    from app.tasks import _load

    tasks = _load()
    rng = _random.Random(12345)
    X_list: list[list[float]] = []
    Y_list: list[list[float]] = []

    for _ in range(n_episodes):
        task = rng.choice(tasks)
        env = TutoringEnv()
        env.reset(task_id=task.id, seed=rng.randint(0, 99999))

        actions = list(ActionType)
        for step_i in range(task.constraints.max_steps):
            if env.is_done:
                break

            at = rng.choice(actions)
            comp_before = dict(env._learner.comprehension)
            att_before = env._learner.attention
            frust_before = env._learner.frustration

            from app.models import Action
            action = Action(action_type=at, rationale="training", payload={})

            try:
                env.step(action)
            except Exception:
                continue

            comp_after = dict(env._learner.comprehension)
            att_after = env._learner.attention
            frust_after = env._learner.frustration

            # Encode input
            x = encode_state(
                action_type=at.value,
                comprehension=comp_before,
                attention=att_before,
                frustration=frust_before,
                confidence=env._learner.confidence,
                fatigue=env._fatigue,
                step=step_i,
                max_steps=task.constraints.max_steps,
                has_assessed=env._coverage.has_assessment,
                has_demo=env._coverage.has_timing_support,
                has_drill=env._coverage.has_micro_drill,
                has_feedback=env._coverage.has_feedback_style,
                has_prerequisite=env._coverage.has_prerequisite,
                learning_style=task.simulation.learning_style.value
                    if hasattr(task.simulation.learning_style, 'value')
                    else str(task.simulation.learning_style),
            )
            X_list.append(x.tolist())

            # Encode target: per-error comp gains + deltas
            y = []
            for et in ERROR_TYPES:
                gain = comp_after.get(et, 0.0) - comp_before.get(et, 0.0)
                y.append(max(0.0, gain))
            y.append(att_after - att_before)
            y.append(frust_after - frust_before)
            Y_list.append(y)

    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(Y_list, dtype=torch.float32)


def train_learner_model(
    n_episodes: int = 3000,
    epochs: int = 100,
    lr: float = 1e-3,
    save_path: str | None = None,
) -> LearnerBehaviorNet:
    """Train the learner behavior model on synthetic data from the simulator."""
    print("[PyTorch] Generating synthetic training data...")
    X, Y = _generate_synthetic_dataset(n_episodes)
    print(f"[PyTorch] Dataset: {X.shape[0]} samples, {X.shape[1]} features -> {Y.shape[1]} targets")

    model = LearnerBehaviorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train/val split
    n = X.shape[0]
    perm = torch.randperm(n)
    split = int(n * 0.85)
    X_train, Y_train = X[perm[:split]], Y[perm[:split]]
    X_val, Y_val = X[perm[split:]], Y[perm[split:]]

    best_val_loss = float("inf")
    best_state = None

    model.train()
    for epoch in range(epochs):
        # Mini-batch training
        batch_size = 256
        perm_train = torch.randperm(X_train.shape[0])
        total_loss = 0.0
        n_batches = 0

        for i in range(0, X_train.shape[0], batch_size):
            idx = perm_train[i:i + batch_size]
            xb, yb = X_train[idx], Y_train[idx]

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = F.mse_loss(val_pred, Y_val).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            avg_train = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1:3d}/{epochs}: train_loss={avg_train:.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    print(f"[PyTorch] Training complete. Best val_loss={best_val_loss:.6f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"[PyTorch] Model saved to {save_path}")

    return model


# ── Model loading ─────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).parent / "sample_data" / "learner_model.pt"
_cached_model: LearnerBehaviorNet | None = None


def get_learner_model() -> LearnerBehaviorNet:
    """Load the pre-trained learner model, training it if weights don't exist."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    model = LearnerBehaviorNet()
    if _MODEL_PATH.exists():
        model.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
    else:
        # Auto-train only if explicitly requested via env var
        import os
        if os.getenv("SIGNADAPT_AUTO_TRAIN", "").strip() in ("1", "true"):
            print("[PyTorch] No pre-trained weights found. Training model...")
            model = train_learner_model(save_path=str(_MODEL_PATH))
        else:
            print("[PyTorch] No pre-trained weights found. Using random initialization.")
            model.eval()

    _cached_model = model
    return model


def predict_gains(
    model: LearnerBehaviorNet,
    action_type: str,
    comprehension: dict[str, float],
    attention: float,
    frustration: float,
    confidence: float,
    fatigue: float,
    step: int,
    max_steps: int,
    has_assessed: bool,
    has_demo: bool,
    has_drill: bool,
    has_feedback: bool,
    has_prerequisite: bool,
    learning_style: str = "mixed",
) -> dict[str, Any]:
    """Predict comprehension gains and state deltas using the neural model."""
    x = encode_state(
        action_type=action_type,
        comprehension=comprehension,
        attention=attention,
        frustration=frustration,
        confidence=confidence,
        fatigue=fatigue,
        step=step,
        max_steps=max_steps,
        has_assessed=has_assessed,
        has_demo=has_demo,
        has_drill=has_drill,
        has_feedback=has_feedback,
        has_prerequisite=has_prerequisite,
        learning_style=learning_style,
    )

    with torch.no_grad():
        pred = model(x.unsqueeze(0)).squeeze(0)

    comp_gains = {et: round(pred[i].item(), 4) for i, et in enumerate(ERROR_TYPES)}
    return {
        "comprehension_gains": comp_gains,
        "attention_delta": round(pred[ERROR_DIM].item(), 4),
        "frustration_delta": round(pred[ERROR_DIM + 1].item(), 4),
    }

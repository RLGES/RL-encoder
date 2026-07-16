"""
training/config.py
~~~~~~~~~~~~~~~~~~
:class:`TrainingConfig` — all hyperparameters for the AlphaZero outer loop.

Design reference: "Detailed Architecture Diagram (Training)" slide

All fields have documented semantics and sensible defaults so the system
can run out-of-the-box for quick smoke tests.  Override via ``dataclasses.replace``
or a YAML/JSON config loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Complete hyperparameter set for EqSat RL training.

    Self-play parameters
    --------------------
    n_games : int
        Number of self-play games to run per training round.
    budget : int
        Maximum rewrite steps per game.
    temperature : float
        Action-sampling temperature from MCTS policy.
        1.0 = proportional; → 0 = greedy.

    MCTS parameters
    ---------------
    num_simulations : int
        Number of MCTS simulations per ``puct()`` call.
    c_puct : float
        Exploration constant in the PUCT formula.

    Training parameters
    -------------------
    K_rounds : int
        Total number of outer training rounds (k = 0..K-1).
    T_steps : int
        Number of gradient update steps per training round.
    batch_size : int
        Mini-batch size drawn from the replay buffer.
    lr : float
        Learning rate for AdamW optimiser.
    weight_decay : float
        L2 regularisation coefficient.
    grad_clip : float
        Gradient norm clipping threshold.

    Architecture parameters
    -----------------------
    embed_dim : int
        Uniform hidden dimension across encoder, policy, value.
    attn_dim : int
        Key/query projection dim inside GNN cross-attention.
    num_gnn_layers : int
        Depth of the Jacobi message-passing stack.
    tau : float
        GNN attention temperature.
    num_rules : int
        Size of the Active Rule Set (determines RuleEmbeddingTable).
    opcode_vocab_size : int, optional
        Number of distinct opcodes; inferred from vocab if None.

    Buffer / checkpoint
    -------------------
    buffer_capacity : int
        Maximum replay buffer size (oldest samples evicted).
    checkpoint_dir : Path
        Directory where checkpoints are saved.
    checkpoint_every : int
        Save a checkpoint every N rounds.
    resume_from : Path, optional
        If set, load weights and resume training from this checkpoint.

    Data
    ----
    training_terms : List[str]
        Pool of source terms used for self-play.  Can be extended at runtime.
    cost_metric : str
        Name of the cost metric (for logging / replay-buffer records).
    """

    # Self-play
    n_games: int = 100
    budget: int = 50
    temperature: float = 1.0

    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.5

    # Training
    K_rounds: int = 20
    T_steps: int = 200
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Architecture
    embed_dim: int = 128
    attn_dim: int = 64
    num_gnn_layers: int = 3
    tau: float = 1.0
    num_rules: int = 64
    opcode_vocab_size: Optional[int] = None

    # Buffer / checkpoint
    buffer_capacity: int = 100_000
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    checkpoint_every: int = 5
    resume_from: Optional[Path] = None

    # Data
    training_terms: List[str] = field(
        default_factory=lambda: [
            "x * 2",
            "x + 0",
            "(a + b) * (a + b)",
            "x * 1 + y * 0",
        ]
    )
    cost_metric: str = "instruction_count"

    # Logging
    log_every: int = 10          # log training stats every N gradient steps
    device: str = "cpu"          # "cuda" for GPU training

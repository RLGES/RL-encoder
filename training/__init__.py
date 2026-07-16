"""
training — AlphaZero-style outer training loop.

Components
----------
  TrainingConfig : dataclass of all hyperparameters
  alpha_zero_loss: combined policy + value loss function
  save / load    : checkpoint utilities
  outer_loop     : top-level training orchestrator

Note: model-heavy imports are lazy to avoid torch_geometric at import time
when only TrainingConfig or loss functions are needed.
"""

from training.config import TrainingConfig
from training.loss import alpha_zero_loss
from training.checkpoint import save_checkpoint, load_checkpoint


def outer_loop(config: "TrainingConfig"):  # type: ignore[name-defined]
    """Lazy import wrapper for outer_loop (avoids eager torch_geometric import)."""
    from training.train_loop import outer_loop as _outer_loop
    return _outer_loop(config)


__all__ = [
    "TrainingConfig",
    "alpha_zero_loss",
    "save_checkpoint",
    "load_checkpoint",
    "outer_loop",
]

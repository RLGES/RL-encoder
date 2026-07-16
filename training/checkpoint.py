"""
training/checkpoint.py
~~~~~~~~~~~~~~~~~~~~~~~
Save and load :class:`~model.alpha_rewrite_net.AlphaRewriteNet` checkpoints.

Also provides :func:`publish_to_inference_server` which atomically swaps
the weights in a live :class:`~data_generation.self_play.InferenceServer`
(needed between training rounds).

Checkpoint format
-----------------
The checkpoint is a Python dict saved with ``torch.save``:

  {
    "round":       int,
    "step":        int,
    "model":       state_dict,
    "optimizer":   state_dict (optional),
    "config":      dict,           (TrainingConfig as dict)
  }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    net: Any,              # AlphaRewriteNet
    round_idx: int,
    step: int,
    optimizer: Optional[Any] = None,
    config: Optional[Any] = None,   # TrainingConfig
) -> None:
    """Serialise *net* weights (and optionally the optimiser state) to *path*.

    Parameters
    ----------
    path : str or Path
        Destination file (e.g. ``checkpoints/theta_k3.pt``).
    net : AlphaRewriteNet
    round_idx : int
        Outer loop round index ``k``.
    step : int
        Total gradient steps taken so far.
    optimizer : torch.optim.Optimizer, optional
    config : TrainingConfig, optional
        If provided, also saved for reproducibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "round": round_idx,
        "step":  step,
        "model": net.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if config is not None:
        from dataclasses import asdict
        payload["config"] = asdict(config)

    torch.save(payload, path)
    logger.info("Checkpoint saved to %s (round=%d, step=%d)", path, round_idx, step)


def load_checkpoint(
    path: str | Path,
    net: Any,                         # AlphaRewriteNet
    optimizer: Optional[Any] = None,  # torch.optim.Optimizer
    device: str = "cpu",
) -> dict:
    """Load a checkpoint from *path* into *net* (and optionally *optimizer*).

    Parameters
    ----------
    path : str or Path
    net : AlphaRewriteNet
    optimizer : Optimizer, optional
    device : str

    Returns
    -------
    dict
        The full checkpoint payload (so callers can restore ``round`` / ``step``).
    """
    path = Path(path)
    payload = torch.load(path, map_location=device)
    net.load_state_dict(payload["model"])
    net.to(device)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    logger.info(
        "Checkpoint loaded from %s (round=%d, step=%d)",
        path, payload.get("round", -1), payload.get("step", -1),
    )
    return payload


def publish_to_inference_server(
    inference_server: Any,  # data_generation.self_play.InferenceServer
    net: Any,               # AlphaRewriteNet
) -> None:
    """Copy current *net* weights into *inference_server*.

    Called by the training loop after each checkpoint to make the latest
    policy available to the self-play workers.

    Parameters
    ----------
    inference_server : InferenceServer
    net : AlphaRewriteNet
    """
    # TODO: make this thread-safe for concurrent training + self-play.
    import copy
    state = copy.deepcopy(net.state_dict())
    inference_server.load(state)
    logger.info("Weights published to InferenceServer.")

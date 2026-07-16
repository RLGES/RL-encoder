"""
training/loss.py
~~~~~~~~~~~~~~~~
AlphaZero-style combined loss function.

Design reference: "Detailed Architecture Diagram (Training)" slide

Loss formula
------------
  L(θ) = MSE(v, z) + CrossEntropy(π_predicted, π_mcts) + λ · ‖θ‖²

where:
  * v         : scalar value prediction from ValueHead, shape [B]
  * z         : actual normalised cost improvement, shape [B]
  * π_pred    : policy logits or probabilities from PolicyHead, shape [B, |A|]
  * π_mcts    : MCTS-improved target policy (soft label), shape [B, |A|]
  * λ         : L2 regularisation weight (optional; can also be handled by the
                optimiser's ``weight_decay`` instead)

Note: when ``π_mcts`` is a hard (one-hot) target, the cross-entropy reduces
to the negative log-probability of the chosen action.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def alpha_zero_loss(
    pi_pred: Tensor,
    pi_target: Tensor,
    v_pred: Tensor,
    z: Tensor,
    l2_reg: float = 0.0,
    model_params: list | None = None,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the AlphaZero combined loss.

    Parameters
    ----------
    pi_pred : FloatTensor  [B, |A|]  or  [|A|]
        Policy logits **or** probabilities from the policy head.
        If values sum to ~1 (probabilities), they are treated as such;
        if not, ``log_softmax`` is applied first.
    pi_target : FloatTensor  [B, |A|]  or  [|A|]
        MCTS-improved target policy (soft labels, sums to 1 per row).
    v_pred : FloatTensor  [B, 1]  or  [B]
        Predicted value from ValueHead.
    z : FloatTensor  [B, 1]  or  [B]
        Target value (actual game outcome / normalised cost improvement).
    l2_reg : float
        Coefficient for optional explicit L2 regularisation term.
        Set to 0 when using AdamW's built-in weight_decay.
    model_params : list, optional
        Iterable of model parameters for L2 term (required if l2_reg > 0).
    eps : float
        Small constant added before ``log`` for numerical stability.

    Returns
    -------
    total_loss  : FloatTensor  scalar
    policy_loss : FloatTensor  scalar
    value_loss  : FloatTensor  scalar

    Notes
    -----
    - ``pi_pred`` and ``pi_target`` must agree on the last dimension.
    - Shapes are broadcast-safe for both single-graph (|A|,) and
      batched ([B, |A|]) inputs.
    """
    pi_pred   = pi_pred.float()
    pi_target = pi_target.float()
    v_pred    = v_pred.float().reshape(-1)
    z         = z.float().reshape(-1)

    # ---- Policy loss: cross-entropy with soft labels --------------------
    # CE = -sum_a(π_target * log π_pred)
    if pi_pred.dim() == 1:
        pi_pred   = pi_pred.unsqueeze(0)    # [1, |A|]
        pi_target = pi_target.unsqueeze(0)  # [1, |A|]

    # Detect whether input is logits or probabilities.
    # Heuristic: if max > 1 or min < 0, treat as logits.
    is_logits = (pi_pred.max() > 1.0) or (pi_pred.min() < 0.0)
    if is_logits:
        log_pi = F.log_softmax(pi_pred, dim=-1)
    else:
        log_pi = torch.log(pi_pred.clamp_min(eps))

    # Normalise target (should already sum to 1, but guard against rounding).
    pi_target = pi_target / pi_target.sum(dim=-1, keepdim=True).clamp_min(eps)

    policy_loss = -(pi_target * log_pi).sum(dim=-1).mean()

    # ---- Value loss: MSE ------------------------------------------------
    value_loss = F.mse_loss(v_pred, z)

    # ---- Optional L2 regularisation term --------------------------------
    l2_loss = torch.tensor(0.0, device=pi_pred.device)
    if l2_reg > 0.0 and model_params is not None:
        for p in model_params:
            l2_loss = l2_loss + p.pow(2).sum()
        l2_loss = l2_reg * l2_loss

    total_loss = policy_loss + value_loss + l2_loss

    return total_loss, policy_loss, value_loss


def policy_cross_entropy(
    log_pi: Tensor,
    pi_target: Tensor,
) -> Tensor:
    """Standalone cross-entropy for the policy (no value term).

    Parameters
    ----------
    log_pi : FloatTensor  [|A|]
        Log-probabilities from :meth:`PolicyHead.logprob`.
    pi_target : FloatTensor  [|A|]
        MCTS visit-count distribution (soft labels).

    Returns
    -------
    FloatTensor  scalar
    """
    eps = 1e-8
    pi_target = pi_target / pi_target.sum().clamp_min(eps)
    return -(pi_target * log_pi).sum()

"""
model/value_head.py
~~~~~~~~~~~~~~~~~~~~
Value head: predicts a scalar state value in ``[-1, +1]`` from the
attention-pooled global graph embedding ``h_G`` produced by
:class:`~egraph_encoder.encoder.EGraphEncoder`.

Design reference: "Detailed Architecture Diagram (EGraphEncoder)" slide

Forward computation
-------------------
  v = tanh( MLP_2layer( h_G ) )

where ``h_G : [B, embed_dim]`` is the graph-level embedding and
``v : [B, 1]`` is the predicted value (one scalar per graph in the batch).

Semantics of *v*
----------------
  * ``v ≈ +1`` : the model predicts significant optimization potential.
  * ``v ≈ -1`` : the model predicts no further improvements are possible
    (terminal or saturated state).
  * Ground-truth target *z* is the actual normalized cost improvement
    achieved from this state to the end of the game.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ValueHead(nn.Module):
    """Two-layer MLP + tanh scalar value predictor.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of ``h_G`` (must match ``EGraphEncoder.embed_dim``).
    hidden_dim : int
        Width of the hidden layer in the 2-layer MLP.

    Forward inputs
    --------------
    h_graph : FloatTensor  [B, embed_dim]
        Global graph embedding from EGraphEncoder.

    Forward outputs
    ---------------
    v : FloatTensor  [B, 1]
        Scalar value prediction, clamped to ``(-1, +1)`` by ``tanh``.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),   # squash to (-1, +1)
        )

    def forward(self, h_graph: Tensor) -> Tensor:
        """Predict value from graph embedding.

        Parameters
        ----------
        h_graph : FloatTensor  [B, embed_dim]

        Returns
        -------
        v : FloatTensor  [B, 1]
            Values in ``(-1, +1)``.
        """
        return self.mlp(h_graph)

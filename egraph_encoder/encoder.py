"""
egraph_encoder/encoder.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Main GNN encoder module: :class:`EGraphEncoder`.

Design reference: "Figure 6.3: RL Encoder Input/Output Structure"

Input
-----
HeteroData with 4 components (see :mod:`egraph_encoder.hetero_data`):
  * ``data["enode"].x``               LongTensor  [N_enodes]
  * ``data["eclass"].x``              FloatTensor [N_eclasses, init_dim]
  * ``("enode","child_of","eclass")`` — child edges + ``.pos``
  * ``("eclass","has_member","enode")``— member edges

Output (3 tensors, returned as a dict)
---------------------------------------
  * ``"h_node"``   : FloatTensor  [N_enodes, embed_dim]
      Per-operation contextualized representations.
  * ``"h_eclass"`` : FloatTensor  [N_eclasses, embed_dim]
      Aggregated per-equivalence-class representations.
  * ``"h_graph"``  : FloatTensor  [B, embed_dim]  (B = batch size, 1 for single graph)
      Attention-pooled global graph embedding used by the value head.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.data import HeteroData  # type: ignore[import]
except ImportError:  # pragma: no cover
    HeteroData = None  # type: ignore[assignment,misc]

try:
    from torch_geometric.utils import scatter, softmax as pyg_softmax  # type: ignore[import]
    _HAS_PYG_SCATTER = True
except ImportError:  # pragma: no cover
    _HAS_PYG_SCATTER = False

from egraph_encoder.gnn_layers import JacobiConvLayer


class EGraphEncoder(nn.Module):
    """Shared heterogeneous GNN encoder over the e-graph.

    Architecture
    ------------
    1. **Embedding layer**: maps integer opcode indices → ``embed_dim`` vectors.
    2. **N Jacobi layers**: alternating enode ↔ eclass message passing.
       See :class:`~egraph_encoder.gnn_layers.JacobiConvLayer` for per-layer
       semantics.
    3. **Attention readout**: produces the scalar-attended graph embedding
       ``h_graph`` via a 1-layer attention pooling over eclass embeddings.

    Parameters
    ----------
    embed_dim : int
        Uniform hidden dimension for all representations.
    attn_dim : int
        Key/query projection dimension within the cross-attention eclass update.
    num_layers : int
        Number of Jacobi message-passing layers (stack depth N).
    tau : float
        Attention temperature — lower = sharper attention.
    opcode_vocab_size : int, optional
        Number of distinct opcodes in the vocabulary.  If ``None``, falls back
        to the shared ``rl_encoder.opcode_vocab`` count.
    pos_vocab_size : int
        Maximum child-slot index.  Determines positional embedding table size.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        attn_dim: int = 64,
        num_layers: int = 3,
        tau: float = 1.0,
        opcode_vocab_size: Optional[int] = None,
        pos_vocab_size: int = 16,
    ) -> None:
        super().__init__()

        if opcode_vocab_size is None:
            try:
                from rl_encoder.opcode_vocab import opcode_vocab_size as _vocab_fn  # type: ignore[import]
                opcode_vocab_size = _vocab_fn()
            except ImportError:  # pragma: no cover
                opcode_vocab_size = 64  # safe default

        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.tau = tau

        # Initial opcode embedding — shared across all layers (skip-injected).
        self.op_embed = nn.Embedding(opcode_vocab_size, embed_dim)

        # Learnable initial e-class representation (broadcast to all eclasses at t=0).
        self.h_cls_init = nn.Parameter(torch.randn(embed_dim))

        # Stack of Jacobi conv layers.
        self.layers = nn.ModuleList([
            JacobiConvLayer(
                embed_dim=embed_dim,
                attn_dim=attn_dim,
                tau=tau,
                pos_vocab_size=pos_vocab_size,
            )
            for _ in range(num_layers)
        ])

        # Graph-level attention readout.
        self.readout_attn = nn.Linear(embed_dim, 1)   # scores each eclass
        self.readout_proj = nn.Sequential(            # final graph embedding projection
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    # ------------------------------------------------------------------

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        """Run the full GNN forward pass.

        Parameters
        ----------
        data : HeteroData
            Produced by :class:`~egraph_encoder.hetero_data.EGraphHeteroData`.

        Returns
        -------
        dict with keys:
          ``"h_node"``   : [N_enodes,  embed_dim]
          ``"h_eclass"`` : [N_eclasses, embed_dim]
          ``"h_graph"``  : [B, embed_dim]
        """
        # ---- unpack graph structure ------------------------------------
        opcode_idx: Tensor = data["enode"].x           # [N_enodes]
        num_enodes   = int(opcode_idx.numel())
        num_eclasses = int(data["eclass"].x.size(0))

        child_store  = data[("enode", "child_of", "eclass")]
        member_store = data[("eclass", "has_member", "enode")]
        child_edge_index  = child_store.edge_index   # [2, E_child]
        child_pos         = child_store.pos          # [E_child]
        member_edge_index = member_store.edge_index  # [2, E_member]

        device = opcode_idx.device

        # ---- initial representations ------------------------------------
        op_features = self.op_embed(opcode_idx)                           # [N_enodes, D]
        h_node   = op_features.clone()                                    # [N_enodes, D]
        h_eclass = self.h_cls_init.unsqueeze(0).expand(num_eclasses, -1)  # [N_eclasses, D]

        # ---- Jacobi message-passing layers ------------------------------
        for layer in self.layers:
            h_node, h_eclass = layer(
                h_node_prev=h_node,
                h_eclass_prev=h_eclass,
                op_features=op_features,           # skip-inject static opcode info
                child_edge_index=child_edge_index,
                child_pos=child_pos,
                member_edge_index=member_edge_index,
            )

        # ---- graph-level attention pooling ------------------------------
        # Determine batch assignment (for mini-batch training).
        if hasattr(data["eclass"], "batch") and data["eclass"].batch is not None:
            eclass_batch: Tensor = data["eclass"].batch
            batch_size = int(eclass_batch.max().item()) + 1
        else:
            eclass_batch = torch.zeros(num_eclasses, dtype=torch.long, device=device)
            batch_size = 1

        # Attention scores over eclasses per graph.
        attn_logits = self.readout_attn(h_eclass).squeeze(-1)  # [N_eclasses]
        attn_scores = self._segment_softmax(attn_logits, eclass_batch, batch_size)

        # Weighted sum → h_graph  [B, embed_dim]
        h_graph = torch.zeros(batch_size, self.embed_dim, device=device)
        h_graph.scatter_add_(
            0,
            eclass_batch.unsqueeze(-1).expand(num_eclasses, self.embed_dim),
            attn_scores.unsqueeze(-1) * h_eclass,
        )
        h_graph = self.readout_proj(h_graph)  # [B, embed_dim]

        # Guard against empty enode tensors (edge case).
        if num_enodes == 0:
            h_node = h_node.reshape(0, self.embed_dim)

        return {
            "h_node":   h_node,    # [N_enodes,   embed_dim]
            "h_eclass": h_eclass,  # [N_eclasses, embed_dim]
            "h_graph":  h_graph,   # [B,           embed_dim]
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_softmax(
        scores: Tensor, segment_ids: Tensor, num_segments: int
    ) -> Tensor:
        """Numerically stable per-segment softmax (same as in gnn_layers)."""
        # TODO: replace with torch_geometric.utils.softmax once confirmed.
        max_val = torch.full((num_segments,), float("-inf"), device=scores.device)
        max_val.scatter_reduce_(0, segment_ids, scores, reduce="amax", include_self=True)
        exp_s = torch.exp(scores - max_val[segment_ids])
        denom = torch.zeros(num_segments, device=scores.device)
        denom.scatter_add_(0, segment_ids, exp_s)
        return exp_s / (denom[segment_ids] + 1e-9)

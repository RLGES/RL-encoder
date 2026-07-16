"""
egraph_encoder/gnn_layers.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Heterogeneous Jacobi-style message-passing layers for the e-graph GNN.

Design reference: "Figure 6.3: RL Encoder Input/Output Structure"

Jacobi-layer semantics (key difference from Gauss-Seidel / standard GNN):
  Both node and e-class updates in a single layer read from the embeddings
  produced at the *previous* layer (frozen context), not from the partially
  updated values of the same layer.  This avoids ordering artifacts and
  matches the parallelism described in the architecture spec.

Layer execution order per iteration k
--------------------------------------
  1. h_node[k]   = NodeUpdate  (reads h_eclass[k-1])
  2. h_eclass[k] = EClassUpdate(reads h_node[k])

Node types and their update mechanisms
---------------------------------------
  * enodes  : Updated by aggregating child e-class embeddings (by operand
              slot), concatenating with the raw op embedding, and passing
              through a 2-layer MLP.
  * eclasses: Updated by multi-head cross-attention over their member enodes,
              with the e-class as query and enodes as keys/values.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.utils import scatter, softmax as pyg_softmax
    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False
    scatter = None        # type: ignore[assignment]
    pyg_softmax = None    # type: ignore[assignment]


class _SumScatter(nn.Module):
    """Minimal fallback segment-sum when PyG is not installed."""

    def forward(self, src: Tensor, index: Tensor, dim_size: int) -> Tensor:
        out = torch.zeros(dim_size, src.size(-1), dtype=src.dtype, device=src.device)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
        return out


class JacobiConvLayer(nn.Module):
    """One Jacobi message-passing iteration over the heterogeneous e-graph.

    Parameters
    ----------
    embed_dim : int
        Uniform hidden dimensionality for enode and eclass representations.
    attn_dim : int
        Dimensionality for query/key projections in the eclass cross-attention.
    tau : float
        Attention temperature (divides raw scores before softmax).
    pos_vocab_size : int
        Number of child-slot positions (operand arity bound).  Determines
        the size of the positional embedding table.

    Forward inputs
    --------------
    h_node_prev : Tensor  [N_enodes,  embed_dim]
        Enode embeddings from the previous Jacobi iteration (or initial op embeds).
    h_eclass_prev : Tensor  [N_eclasses, embed_dim]
        E-class embeddings from the previous Jacobi iteration.
    op_features : Tensor  [N_enodes, embed_dim]
        Static opcode embeddings (injected at every layer, skip-connection style).
    child_edge_index : LongTensor  [2, E_child]
        ``[0]`` = enode indices, ``[1]`` = eclass indices.
    child_pos : LongTensor  [E_child]
        Slot position of each child edge (used for positional embedding).
    member_edge_index : LongTensor  [2, E_member]
        ``[0]`` = eclass indices (queries), ``[1]`` = enode indices (keys/values).

    Forward outputs
    ---------------
    h_node_new : Tensor  [N_enodes,  embed_dim]
    h_eclass_new : Tensor  [N_eclasses, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int = 128,
        attn_dim: int = 64,
        tau: float = 1.0,
        pos_vocab_size: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.tau = tau

        # ---- enode update components ----
        self.pos_embed = nn.Embedding(pos_vocab_size, embed_dim)
        self.mlp_child = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mlp_node = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ---- eclass update components (cross-attention) ----
        self.W_q = nn.Linear(embed_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, attn_dim, bias=False)
        self.ln_q = nn.LayerNorm(attn_dim)
        self.ln_k = nn.LayerNorm(attn_dim)
        self.proj_eclass = nn.Linear(attn_dim, embed_dim, bias=False)

        self._scatter_fn = _SumScatter()  # fallback; overridden at forward if PyG present

    # ------------------------------------------------------------------

    def forward(
        self,
        h_node_prev: Tensor,
        h_eclass_prev: Tensor,
        op_features: Tensor,
        child_edge_index: Tensor,
        child_pos: Tensor,
        member_edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run one Jacobi iteration.

        See class docstring for tensor shapes.
        """
        h_node_new = self._update_enodes(
            h_eclass_prev, op_features, child_edge_index, child_pos
        )
        h_eclass_new = self._update_eclasses(
            h_eclass_prev, h_node_new, member_edge_index
        )
        return h_node_new, h_eclass_new

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_enodes(
        self,
        h_eclass_prev: Tensor,
        op_features: Tensor,
        child_edge_index: Tensor,
        child_pos: Tensor,
    ) -> Tensor:
        """Aggregate child e-class messages into per-enode embeddings.

        Shapes
        ------
        h_eclass_prev : [N_eclasses, embed_dim]
        op_features   : [N_enodes,   embed_dim]
        child_edge_index : [2, E_child]   row0=enode, row1=eclass
        child_pos     : [E_child]

        Returns
        -------
        h_node : [N_enodes, embed_dim]
        """
        num_enodes = op_features.size(0)
        device = op_features.device

        if child_edge_index.numel() == 0:
            child_agg = torch.zeros(num_enodes, self.embed_dim, device=device)
        else:
            enode_idx = child_edge_index[0]   # [E_child]
            eclass_idx = child_edge_index[1]  # [E_child]
            pos = child_pos.clamp(0, self.pos_embed.num_embeddings - 1)

            # child message = MLP(h_eclass + pos_embed)
            child_input = h_eclass_prev[eclass_idx] + self.pos_embed(pos)
            child_msg = self.mlp_child(child_input)           # [E_child, D]

            # sum-aggregate per enode
            # TODO: replace with scatter from PyG for GPU efficiency.
            child_agg = torch.zeros(num_enodes, self.embed_dim, device=device)
            child_agg.scatter_add_(
                0, enode_idx.unsqueeze(-1).expand_as(child_msg), child_msg
            )

        h_node = self.mlp_node(torch.cat([op_features, child_agg], dim=-1))
        return h_node

    def _update_eclasses(
        self,
        h_eclass_prev: Tensor,
        h_node_new: Tensor,
        member_edge_index: Tensor,
    ) -> Tensor:
        """Cross-attend over member enodes to update e-class embeddings.

        Shapes
        ------
        h_eclass_prev : [N_eclasses, embed_dim]  (queries)
        h_node_new    : [N_enodes,   embed_dim]  (keys + values)
        member_edge_index : [2, E_member]  row0=eclass, row1=enode

        Returns
        -------
        h_eclass : [N_eclasses, embed_dim]
        """
        num_eclasses = h_eclass_prev.size(0)
        device = h_eclass_prev.device

        if member_edge_index.numel() == 0:
            return h_eclass_prev  # nothing to attend over

        eclass_idx = member_edge_index[0]  # [E_member]
        enode_idx  = member_edge_index[1]  # [E_member]

        # Projected queries from e-class side; keys+values from enode side.
        q = self.ln_q(self.W_q(h_eclass_prev))   # [N_eclasses, attn_dim]
        k = self.ln_k(self.W_k(h_node_new))       # [N_enodes,   attn_dim]
        v = self.W_v(h_node_new)                   # [N_enodes,   attn_dim]

        q_e = q[eclass_idx]   # [E_member, attn_dim]
        k_e = k[enode_idx]    # [E_member, attn_dim]
        v_e = v[enode_idx]    # [E_member, attn_dim]

        score = (q_e * k_e).sum(dim=-1) / (math.sqrt(self.attn_dim) * self.tau)
        # [E_member] — softmax over edges that share the same eclass (query)

        # TODO: use torch_geometric.utils.softmax for sparse-group softmax.
        #       Currently using a manual segment-softmax fallback.
        alpha = self._segment_softmax(score, eclass_idx, num_eclasses)

        h_agg = torch.zeros(num_eclasses, self.attn_dim, device=device)
        h_agg.scatter_add_(
            0,
            eclass_idx.unsqueeze(-1).expand_as(v_e),
            alpha.unsqueeze(-1) * v_e,
        )

        h_eclass_new = self.proj_eclass(h_agg)  # [N_eclasses, embed_dim]

        # Keep previous embeddings for any e-class with no members.
        counts = torch.zeros(num_eclasses, device=device)
        counts.scatter_add_(0, eclass_idx, torch.ones(eclass_idx.size(0), device=device))
        no_member = counts == 0
        if no_member.any():
            h_eclass_new = h_eclass_new.clone()
            h_eclass_new[no_member] = h_eclass_prev[no_member]

        return h_eclass_new

    @staticmethod
    def _segment_softmax(scores: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
        """Numerically stable softmax within each segment (query group).

        Parameters
        ----------
        scores : Tensor  [E]
        segment_ids : LongTensor  [E]  — which query each edge belongs to
        num_segments : int

        Returns
        -------
        alpha : Tensor  [E]
        """
        # TODO: replace with PyG's softmax() for GPU-kernel efficiency.
        # Max per segment (for numerical stability).
        max_val = torch.full((num_segments,), float("-inf"), device=scores.device)
        max_val.scatter_reduce_(0, segment_ids, scores, reduce="amax", include_self=True)
        scores_shifted = scores - max_val[segment_ids]
        exp_scores = torch.exp(scores_shifted)
        denom = torch.zeros(num_segments, device=scores.device)
        denom.scatter_add_(0, segment_ids, exp_scores)
        return exp_scores / (denom[segment_ids] + 1e-9)

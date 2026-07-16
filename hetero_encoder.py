"""
DEPRECATED: This file is a legacy placeholder. 
Please use the new modular implementation in `egraph_encoder/encoder.py`.

Original implementation follows below for backward compatibility:
Heterogeneous e-graph encoder for RL policy/value networks.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter, softmax

from rl_encoder.opcode_vocab import opcode_vocab_size as default_opcode_vocab_size


class EGraphEncoder(nn.Module):
    """Encode batched e-graphs into node, eclass, and graph embeddings."""

    def __init__(
        self,
        embed_dim: int = 128,
        attn_dim: int = 64,
        num_layers: int = 3,
        tau: float = 1.0,
        opcode_vocab_size: int | None = None,
    ):
        super().__init__()
        if opcode_vocab_size is None:
            opcode_vocab_size = default_opcode_vocab_size()

        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.tau = tau

        self.op_embed = nn.Embedding(opcode_vocab_size, embed_dim)
        self.h_cls = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Embedding(16, embed_dim)

        self.mlp_child = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp_node = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.W_q = nn.ModuleList(
            [nn.Linear(embed_dim, attn_dim, bias=False) for _ in range(num_layers)]
        )
        self.W_k = nn.ModuleList(
            [nn.Linear(embed_dim, attn_dim, bias=False) for _ in range(num_layers)]
        )
        self.W_v = nn.ModuleList(
            [nn.Linear(embed_dim, attn_dim, bias=False) for _ in range(num_layers)]
        )
        self.layer_norm_q = nn.ModuleList([nn.LayerNorm(attn_dim) for _ in range(num_layers)])
        self.layer_norm_k = nn.ModuleList([nn.LayerNorm(attn_dim) for _ in range(num_layers)])

        # Attention values are attn_dim; project back so h_eclass stays embed_dim across layers.
        self.eclass_proj = nn.ModuleList(
            [nn.Linear(attn_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )

        self.mlp_readout = nn.Linear(embed_dim, 1)
        self.mlp_graph = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
        )

    def _node_update(
        self,
        layer: int,
        op_features: Tensor,
        h_eclass_prev: Tensor,
        child_edge_index: Tensor,
        child_pos: Tensor,
    ) -> Tensor:
        num_enodes = op_features.size(0)
        device = op_features.device

        if child_edge_index.numel() == 0:
            child_sum = torch.zeros((num_enodes, self.embed_dim), device=device)
        else:
            src = child_edge_index[0]
            dst = child_edge_index[1]
            pos = child_pos.clamp(min=0, max=15)

            child_input = h_eclass_prev[dst] + self.pos_embed(pos)
            child_msg = self.mlp_child[layer](child_input)
            child_sum = scatter(child_msg, src, dim=0, dim_size=num_enodes, reduce="sum")

        node_input = torch.cat([op_features, child_sum], dim=-1)
        return self.mlp_node[layer](node_input)

    def _eclass_update(
        self,
        layer: int,
        h_eclass_prev: Tensor,
        h_node_new: Tensor,
        member_edge_index: Tensor,
    ) -> Tensor:
        num_eclasses = h_eclass_prev.size(0)
        device = h_eclass_prev.device

        if member_edge_index.numel() == 0:
            return h_eclass_prev

        src = member_edge_index[0]
        dst = member_edge_index[1]

        q = self.layer_norm_q[layer](self.W_q[layer](h_eclass_prev))
        k = self.layer_norm_k[layer](self.W_k[layer](h_node_new))
        v = self.W_v[layer](h_node_new)

        q_edge = q[src]
        k_edge = k[dst]
        v_edge = v[dst]

        score = (q_edge * k_edge).sum(dim=-1) / math.sqrt(self.attn_dim)
        alpha = softmax(score / self.tau, src)

        weighted_v = alpha.unsqueeze(-1) * v_edge
        h_attn = scatter(weighted_v, src, dim=0, dim_size=num_eclasses, reduce="sum")
        h_eclass_new = self.eclass_proj[layer](h_attn)

        counts = scatter(
            torch.ones((src.numel(),), dtype=h_eclass_prev.dtype, device=device),
            src,
            dim=0,
            dim_size=num_eclasses,
            reduce="sum",
        )
        no_member_mask = counts == 0
        if no_member_mask.any():
            h_eclass_new = h_eclass_new.clone()
            h_eclass_new[no_member_mask] = h_eclass_prev[no_member_mask]

        return h_eclass_new

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        """Run Jacobi-style message passing and graph readout."""
        opcode_idx = data["enode"].x
        num_enodes = int(opcode_idx.numel())
        num_eclasses = int(data["eclass"].x.size(0))

        child_store = data[("enode", "child_of", "eclass")]
        member_store = data[("eclass", "has_member", "enode")]
        child_edge_index = child_store.edge_index
        child_pos = child_store.pos
        member_edge_index = member_store.edge_index

        op_features = self.op_embed(opcode_idx)
        h_node = op_features
        h_eclass = self.h_cls.unsqueeze(0).expand(num_eclasses, -1)

        for k in range(self.num_layers):
            h_eclass_prev = h_eclass

            # Jacobi: node update reads h_eclass from previous iteration.
            h_node_new = self._node_update(
                layer=k,
                op_features=op_features,
                h_eclass_prev=h_eclass_prev,
                child_edge_index=child_edge_index,
                child_pos=child_pos,
            )

            h_eclass_new = self._eclass_update(
                layer=k,
                h_eclass_prev=h_eclass_prev,
                h_node_new=h_node_new,
                member_edge_index=member_edge_index,
            )

            h_node = h_node_new
            h_eclass = h_eclass_new

        if hasattr(data["eclass"], "batch"):
            eclass_batch = data["eclass"].batch
            batch_size = int(eclass_batch.max().item() + 1) if eclass_batch.numel() > 0 else 1
        else:
            eclass_batch = torch.zeros((num_eclasses,), dtype=torch.long, device=h_eclass.device)
            batch_size = 1

        readout_logits = self.mlp_readout(h_eclass).squeeze(-1)
        beta = softmax(readout_logits, eclass_batch)
        h_attn = scatter(
            beta.unsqueeze(-1) * h_eclass,
            eclass_batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )

        root_idx = getattr(data["eclass"], "root_idx", None)
        if root_idx is None:
            warnings.warn(
                "No root_idx provided; falling back to attention pooled embedding only.",
                stacklevel=2,
            )
            h_root = h_attn
        else:
            if not isinstance(root_idx, torch.Tensor):
                root_idx = torch.tensor(root_idx, dtype=torch.long, device=h_eclass.device)
            root_idx = root_idx.to(h_eclass.device)
            if root_idx.numel() == 1 and batch_size > 1:
                root_idx = root_idx.expand(batch_size)

            if root_idx.numel() != batch_size:
                warnings.warn(
                    "root_idx shape mismatch with batch size; falling back to attention pooled embedding only.",
                    stacklevel=2,
                )
                h_root = h_attn
            else:
                h_root = torch.zeros_like(h_attn)
                valid = root_idx >= 0
                if valid.any():
                    h_root[valid] = h_eclass[root_idx[valid]]
                if (~valid).any():
                    warnings.warn(
                        "Some graphs have no valid root_idx; using attention pooled embedding for those graphs.",
                        stacklevel=2,
                    )
                    h_root[~valid] = h_attn[~valid]

        h_graph = self.mlp_graph(torch.cat([h_root, h_attn], dim=-1))

        if num_enodes == 0:
            h_node = h_node.reshape(0, self.embed_dim)

        return {
            "h_node": h_node,
            "h_eclass": h_eclass,
            "h_graph": h_graph,
        }

"""Helper functions for RL encoder integration and training utilities."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, HeteroData

from egraph_bridge.simple_egraph import EGraph
from rewrite_rules.rule_base import RewriteRule
from rl_encoder.egraph_dataset import egraph_to_heterodata


class _RuleInitMLP(nn.Module):
    """Deterministic tiny MLP for feature-based rule embedding initialization."""

    def __init__(self, embed_dim: int):
        super().__init__()
        hidden = max(16, embed_dim // 2)
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.act = nn.ReLU()

        with torch.no_grad():
            self.fc1.weight.fill_(0.05)
            self.fc1.bias.zero_()
            self.fc2.weight.fill_(0.05)
            self.fc2.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


_RULE_INIT_MLPS: dict[int, _RuleInitMLP] = {}


def _get_rule_init_mlp(embed_dim: int) -> _RuleInitMLP:
    if embed_dim not in _RULE_INIT_MLPS:
        _RULE_INIT_MLPS[embed_dim] = _RuleInitMLP(embed_dim)
    return _RULE_INIT_MLPS[embed_dim]


def rule_init_from_features(rule: RewriteRule, embed_dim: int) -> Tensor:
    """Create feature-based initialization vector for a new rewrite rule embedding."""
    lhs_len = float(len(rule.lhs))
    rhs_len = float(len(rule.rhs))
    max_arity = float(max((len(p.srcs) for p in rule.lhs), default=0))
    distinct_lhs_ops = float(len({p.opcode for p in rule.lhs}))

    features = torch.tensor(
        [lhs_len, rhs_len, max_arity, distinct_lhs_ops],
        dtype=torch.float,
    ).unsqueeze(0)
    mlp = _get_rule_init_mlp(embed_dim)
    with torch.no_grad():
        return mlp(features).squeeze(0)


def batch_egraphs(
    egraph_list: List[EGraph],
    root_ids: List[Optional[int]],
) -> HeteroData:
    """Convert and batch multiple EGraphs into one HeteroData batch."""
    if len(egraph_list) != len(root_ids):
        raise ValueError("egraph_list and root_ids must have the same length")

    data_list = [
        egraph_to_heterodata(egraph=egraph, root_eclass_id=root_id)
        for egraph, root_id in zip(egraph_list, root_ids)
    ]

    batch = Batch.from_data_list(data_list)

    # Build global root indices per graph to remain correct under batching.
    root_global: List[int] = []
    offset = 0
    for data in data_list:
        eclass_count = int(data["eclass"].x.size(0))
        root_idx_local = getattr(data["eclass"], "root_idx", None)
        if root_idx_local is None:
            root_global.append(-1)
        else:
            root_global.append(int(root_idx_local.item()) + offset)
        offset += eclass_count

    batch["eclass"].root_idx = torch.tensor(root_global, dtype=torch.long)
    return batch


def compute_reward(cost_before: int, cost_after: int) -> float:
    """Compute normalized cost-improvement reward clipped to [-1, 1]."""
    if cost_before <= 0:
        return 0.0

    raw = float(cost_before - cost_after) / float(cost_before)
    return max(-1.0, min(1.0, raw))

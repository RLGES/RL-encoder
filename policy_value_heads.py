"""Policy and value heads for AlphaZero-style e-graph control."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from rl_encoder.hetero_encoder import EGraphEncoder
from rl_encoder.opcode_vocab import opcode_vocab_size
from rl_encoder.opcode_vocab import rule_vocab_size as default_rule_vocab_size


class PolicyHead(nn.Module):
    """Scores valid (rule, eclass) actions and normalizes over valid set only."""

    def __init__(self, embed_dim: int = 128, rule_vocab_size: Optional[int] = None):
        super().__init__()
        if rule_vocab_size is None:
            rule_vocab_size = max(default_rule_vocab_size(), 1)

        self.rule_embed = nn.Embedding(rule_vocab_size, embed_dim)
        self.mlp_policy = nn.Sequential(
            nn.Linear(2 * embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, h_eclass: Tensor, valid_actions: List[Tuple[int, int]]) -> Tensor:
        """Return probability distribution over valid actions only."""
        if len(valid_actions) == 0:
            return torch.empty((0,), dtype=h_eclass.dtype, device=h_eclass.device)

        rule_idx = torch.tensor(
            [a[0] for a in valid_actions], dtype=torch.long, device=h_eclass.device
        )
        eclass_idx = torch.tensor(
            [a[1] for a in valid_actions], dtype=torch.long, device=h_eclass.device
        )

        action_repr = torch.cat([h_eclass[eclass_idx], self.rule_embed(rule_idx)], dim=-1)
        logits = self.mlp_policy(action_repr).squeeze(-1)
        return torch.softmax(logits, dim=0)


class ValueHead(nn.Module):
    """Predict scalar value in [-1, 1] from graph embedding."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.mlp_value = nn.Sequential(
            nn.Linear(2 * embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, h_graph: Tensor) -> Tensor:
        """Return value prediction per graph in batch."""
        return self.mlp_value(h_graph)


class PolicyValueNetwork(nn.Module):
    """Shared e-graph encoder with separate policy and value heads."""

    def __init__(
        self,
        embed_dim: int = 128,
        attn_dim: int = 64,
        num_layers: int = 3,
        tau: float = 1.0,
        opcode_vocab_n: Optional[int] = None,
        rule_vocab_n: Optional[int] = None,
    ):
        super().__init__()

        if opcode_vocab_n is None:
            opcode_vocab_n = opcode_vocab_size()
        if rule_vocab_n is None:
            rule_vocab_n = max(default_rule_vocab_size(), 1)

        self.encoder = EGraphEncoder(
            embed_dim=embed_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            tau=tau,
            opcode_vocab_size=opcode_vocab_n,
        )
        self.policy_head = PolicyHead(embed_dim=embed_dim, rule_vocab_size=rule_vocab_n)
        self.value_head = ValueHead(embed_dim=embed_dim)

    def forward(
        self,
        data: HeteroData,
        valid_actions: List[Tuple[int, int]],
    ) -> Tuple[Tensor, Tensor]:
        """Return policy over valid actions and value predictions."""
        enc_out = self.encoder(data)
        h_eclass = enc_out["h_eclass"]
        h_graph = enc_out["h_graph"]

        pi = self.policy_head(h_eclass, valid_actions)
        v = self.value_head(h_graph)
        return pi, v

    @staticmethod
    def compute_loss(
        pi_pred: Tensor,
        pi_target: Tensor,
        v_pred: Tensor,
        v_target: Tensor,
    ) -> Tensor:
        """Compute CE(policy) + MSE(value) objective."""
        eps = 1e-8

        if pi_pred.numel() == 0:
            policy_loss = torch.tensor(0.0, device=v_pred.device, dtype=v_pred.dtype)
        elif pi_target.dtype in (torch.int32, torch.int64):
            log_pi = torch.log(pi_pred.clamp_min(eps)).unsqueeze(0)
            policy_loss = F.nll_loss(log_pi, pi_target.view(-1))
        else:
            policy_target = pi_target.to(pi_pred.dtype)
            policy_target = policy_target / policy_target.sum().clamp_min(eps)
            policy_loss = -(policy_target * torch.log(pi_pred.clamp_min(eps))).sum()

        value_loss = F.mse_loss(v_pred.view_as(v_target), v_target)
        return policy_loss + value_loss

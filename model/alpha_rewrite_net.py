"""
model/alpha_rewrite_net.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Top-level network module: :class:`AlphaRewriteNet`.

Wires together:
  * :class:`~egraph_encoder.encoder.EGraphEncoder`
  * :class:`~model.rule_embeddings.RuleEmbeddingTable`
  * :class:`~model.policy_head.PolicyHead`
  * :class:`~model.value_head.ValueHead`

into a single ``nn.Module`` whose ``forward`` maps an e-graph state to a
(policy, value) pair, exactly as required by the MCTS inference server.

Design reference: "Detailed Architecture Diagram (EGraphEncoder)" slide

Forward contract
----------------
Input:
  * ``hetero_data``        : HeteroData produced by EGraphHeteroData.build()
  * ``action_rule_ids``    : LongTensor  [|A|]  — rule index per candidate action
  * ``action_eclass_ids``  : LongTensor  [|A|]  — eclass index per candidate action
  * ``legal_action_mask``  : BoolTensor  [|A|]  — True = legal

Output:
  * ``pi`` : FloatTensor  [|A|]  — probability distribution over actions
  * ``v``  : FloatTensor  [1]    — scalar value in (-1, +1)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.data import HeteroData  # type: ignore[import]
except ImportError:  # pragma: no cover
    HeteroData = None  # type: ignore[assignment,misc]

from egraph_encoder.encoder import EGraphEncoder
from model.policy_head import PolicyHead
from model.rule_embeddings import RuleEmbeddingTable
from model.value_head import ValueHead


class AlphaRewriteNet(nn.Module):
    """Full AlphaZero-style network for the e-graph rewriting task.

    Parameters
    ----------
    embed_dim : int
        Shared embedding dimensionality across all sub-modules.
    attn_dim : int
        Key/query projection dim inside EGraphEncoder.
    num_gnn_layers : int
        Stack depth of Jacobi message-passing layers.
    tau : float
        GNN attention temperature.
    num_rules : int
        Size of the active rule set (determines RuleEmbeddingTable size).
    opcode_vocab_size : int, optional
        Number of distinct opcodes; inferred from ``rl_encoder.opcode_vocab``
        if ``None``.
    policy_hidden_dim : int
        Hidden width of the PolicyHead MLP.
    value_hidden_dim : int
        Hidden width of the ValueHead MLP.

    Notes
    -----
    All three sub-networks share ``embed_dim`` as their common interface
    dimension, ensuring the output of the encoder slots directly into both
    heads without any additional projection layers.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        attn_dim: int = 64,
        num_gnn_layers: int = 3,
        tau: float = 1.0,
        num_rules: int = 64,
        opcode_vocab_size: int | None = None,
        policy_hidden_dim: int = 256,
        value_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = EGraphEncoder(
            embed_dim=embed_dim,
            attn_dim=attn_dim,
            num_layers=num_gnn_layers,
            tau=tau,
            opcode_vocab_size=opcode_vocab_size,
        )
        self.rule_table = RuleEmbeddingTable(
            num_rules=num_rules,
            embed_dim=embed_dim,
        )
        self.policy_head = PolicyHead(
            embed_dim=embed_dim,
            hidden_dim=policy_hidden_dim,
        )
        self.value_head = ValueHead(
            embed_dim=embed_dim,
            hidden_dim=value_hidden_dim,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        hetero_data: HeteroData,
        action_rule_ids: Tensor,
        action_eclass_ids: Tensor,
        legal_action_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run the full forward pass.

        Parameters
        ----------
        hetero_data : HeteroData
            E-graph state as a heterogeneous graph (from EGraphHeteroData).
        action_rule_ids : LongTensor  [|A|]
            Rule index for each candidate action.
        action_eclass_ids : LongTensor  [|A|]
            E-class index (row into h_eclass) for each candidate action.
        legal_action_mask : BoolTensor  [|A|]
            True where the action is currently applicable.

        Returns
        -------
        pi : FloatTensor  [|A|]
            Masked softmax policy distribution.
        v  : FloatTensor  [1]
            Scalar value prediction (tanh-squashed).
        """
        # ---- 1. Encode the e-graph state --------------------------------
        enc_out = self.encoder(hetero_data)
        h_eclass: Tensor = enc_out["h_eclass"]  # [N_eclasses, embed_dim]
        h_graph:  Tensor = enc_out["h_graph"]   # [B,           embed_dim]

        # ---- 2. Policy head ---------------------------------------------
        rule_embeds = self.rule_table(action_rule_ids)             # [|A|, embed_dim]
        eclass_embeds_at_actions = h_eclass[action_eclass_ids]     # [|A|, embed_dim]

        pi, _ = self.policy_head(
            rule_embeds=rule_embeds,
            eclass_embeds_at_actions=eclass_embeds_at_actions,
            legal_action_mask=legal_action_mask,
        )

        # ---- 3. Value head ----------------------------------------------
        v = self.value_head(h_graph)  # [B, 1]

        # For single-graph inference, squeeze batch dim → scalar.
        v = v.squeeze(0)  # [1]  (or [B, 1] in batched mode — TODO: handle both)

        return pi, v

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        hetero_data: HeteroData,
        action_rule_ids: Tensor,
        action_eclass_ids: Tensor,
        legal_action_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Inference-mode forward (no grad, eval semantics).

        Returns same shapes as :meth:`forward`.
        """
        self.eval()
        return self.forward(
            hetero_data=hetero_data,
            action_rule_ids=action_rule_ids,
            action_eclass_ids=action_eclass_ids,
            legal_action_mask=legal_action_mask,
        )

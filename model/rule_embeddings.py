"""
model/rule_embeddings.py
~~~~~~~~~~~~~~~~~~~~~~~~
Learnable embedding table indexed by verified rewrite-rule ID.

Each verified rule in the Active Rule Set produced by Learn-R gets a unique
integer ID (registered via ``rl_encoder.opcode_vocab.register_rule``).  This
module keeps a standard ``nn.Embedding`` over those IDs so the policy head
can look up a dense representation for any candidate rule.

Usage
-----
>>> table = RuleEmbeddingTable(num_rules=50, embed_dim=128)
>>> rule_idx = torch.tensor([3, 7, 12], dtype=torch.long)
>>> embeds = table(rule_idx)    # shape [3, 128]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RuleEmbeddingTable(nn.Module):
    """Learnable embedding table: rule_id → embed_dim vector.

    Parameters
    ----------
    num_rules : int
        Total number of distinct verified rewrite rules (Active Rule Set size).
        This can be grown dynamically; call :meth:`expand` when new rules are
        discovered by Learn-R during training.
    embed_dim : int
        Embedding dimensionality; must match ``EGraphEncoder.embed_dim``.
    padding_idx : int, optional
        If set, the embedding at this index is kept all-zeros (useful for
        masking invalid / not-yet-learned rules).

    Forward inputs
    --------------
    rule_idx : LongTensor  [K]
        Indices of the K candidate rules to embed.

    Forward outputs
    ---------------
    embeddings : FloatTensor  [K, embed_dim]
    """

    def __init__(
        self,
        num_rules: int,
        embed_dim: int = 128,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.table = nn.Embedding(
            num_embeddings=num_rules,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, rule_idx: Tensor) -> Tensor:
        """Look up rule embeddings.

        Parameters
        ----------
        rule_idx : LongTensor  [K]

        Returns
        -------
        FloatTensor  [K, embed_dim]
        """
        return self.table(rule_idx)

    def expand(self, new_num_rules: int) -> None:
        """Grow the table to ``new_num_rules`` without losing existing weights.

        Call when Learn-R discovers additional verified rules mid-training.

        Parameters
        ----------
        new_num_rules : int
            New total number of rules.  Must be > current table size.
        """
        # TODO: implement parameter re-initialization preserving old weights.
        old_size = self.table.num_embeddings
        if new_num_rules <= old_size:
            return
        old_weight = self.table.weight.data
        new_weight = torch.zeros(
            new_num_rules, self.embed_dim, dtype=old_weight.dtype, device=old_weight.device
        )
        new_weight[:old_size] = old_weight
        nn.init.normal_(new_weight[old_size:], mean=0.0, std=0.02)
        self.table = nn.Embedding.from_pretrained(new_weight, freeze=False)

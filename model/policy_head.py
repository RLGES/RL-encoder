r"""
model/policy_head.py
~~~~~~~~~~~~~~~~~~~~~
Policy head: computes a masked probability distribution over the action space
``A = { (rule, eclass) }``.


Design reference: "Detailed Architecture Diagram (EGraphEncoder)" slide

Action definition
-----------------
Each action is a pair ``(rule_idx, eclass_idx)`` meaning "apply rule
``rule_idx`` to equivalence class ``eclass_idx``".

Forward computation (per target e-class *i* or jointly over all valid actions)
-------------------------------------------------------------------------------
  1. ``rule_embed = RuleEmbeddingTable[rule_idx]``      [|A|, D]
  2. ``H_i       = eclass_embeds[eclass_idx]``          [|A|, D]
  3. ``x         = concat([rule_embed, H_i])``          [|A|, 2D]
  4. ``logits z  = MLP_2layer(x)``                      [|A|]
  5. ``masked     = z.masked_fill(~legal_mask, -inf)``  [|A|]
  6. ``pi         = softmax(masked)``                   [|A|]  ∈ Δ^{|A|}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PolicyHead(nn.Module):
    """Action-logit scorer and masked softmax policy.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of rule and e-class embeddings (both must be ``embed_dim``).
    hidden_dim : int
        Hidden layer width in the 2-layer MLP.

    Forward inputs
    --------------
    rule_embeds : FloatTensor  [|A|, embed_dim]
        Rule embedding for each candidate action (from RuleEmbeddingTable).
    eclass_embeds_at_actions : FloatTensor  [|A|, embed_dim]
        The e-class embedding H_i for the target e-class of each action.
    legal_action_mask : BoolTensor  [|A|]
        ``True`` for legal actions; illegal actions are set to ``-inf`` before
        softmax to ensure they receive zero probability.

    Forward outputs
    ---------------
    pi : FloatTensor  [|A|]
        Probability simplex over the ``|A|`` candidate actions.
    logits : FloatTensor  [|A|]
        Pre-softmax logits (useful for cross-entropy loss computation).
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # 2-layer MLP: concat(rule_embed, H_i) → scalar logit
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        rule_embeds: Tensor,
        eclass_embeds_at_actions: Tensor,
        legal_action_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute masked policy distribution.

        Parameters
        ----------
        rule_embeds : FloatTensor  [|A|, embed_dim]
        eclass_embeds_at_actions : FloatTensor  [|A|, embed_dim]
        legal_action_mask : BoolTensor  [|A|]

        Returns
        -------
        pi : FloatTensor  [|A|]
        logits : FloatTensor  [|A|]
        """
        if rule_embeds.size(0) == 0:
            empty = torch.empty(0, device=rule_embeds.device)
            return empty, empty

        # Step 3-4: concat and score
        x = torch.cat([rule_embeds, eclass_embeds_at_actions], dim=-1)  # [|A|, 2D]
        logits = self.mlp(x).squeeze(-1)                                  # [|A|]

        # Step 5: mask illegal actions
        masked_logits = logits.masked_fill(~legal_action_mask, float("-inf"))

        # Step 6: softmax
        pi = torch.softmax(masked_logits, dim=0)  # [|A|]

        return pi, logits

    # ------------------------------------------------------------------

    def logprob(
        self,
        rule_embeds: Tensor,
        eclass_embeds_at_actions: Tensor,
        legal_action_mask: Tensor,
    ) -> Tensor:
        """Return log-probabilities (for cross-entropy loss).

        Returns
        -------
        log_pi : FloatTensor  [|A|]
        """
        _, logits = self.forward(rule_embeds, eclass_embeds_at_actions, legal_action_mask)
        masked = logits.masked_fill(~legal_action_mask, float("-inf"))
        return F.log_softmax(masked, dim=0)

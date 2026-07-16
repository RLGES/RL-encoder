"""
model — Policy head, value head, rule embeddings, and the top-level
AlphaRewriteNet that wires them all together.

Design reference: "Detailed Architecture Diagram (EGraphEncoder)" slide

Public API
----------
  AlphaRewriteNet   : full network (encoder + policy + value)
  PolicyHead        : action logits over the (rule, eclass) action space
  ValueHead         : scalar state-value prediction
  RuleEmbeddingTable: learnable embedding per verified rewrite rule
"""

from model.policy_head import PolicyHead
from model.value_head import ValueHead
from model.rule_embeddings import RuleEmbeddingTable
from model.alpha_rewrite_net import AlphaRewriteNet

__all__ = [
    "AlphaRewriteNet",
    "PolicyHead",
    "ValueHead",
    "RuleEmbeddingTable",
]

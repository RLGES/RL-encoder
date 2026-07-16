"""
tests/test_model.py
~~~~~~~~~~~~~~~~~~~~
Smoke tests for Module 2: PolicyHead, ValueHead, RuleEmbeddingTable,
and AlphaRewriteNet.

Tests
-----
* test_policy_head_shapes   : correct output shape [|A|]
* test_policy_sums_to_one   : policy probabilities sum to 1
* test_policy_mask          : illegal actions get zero probability
* test_value_head_shapes    : output is [B, 1] in (-1, 1)
* test_rule_embedding_shapes: table lookup returns [K, embed_dim]
* test_alpha_rewrite_net    : full forward pass shape check
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from model.alpha_rewrite_net import AlphaRewriteNet
from model.policy_head import PolicyHead
from model.rule_embeddings import RuleEmbeddingTable
from model.value_head import ValueHead


class TestPolicyHead:
    def test_output_shapes(self, embed_dim: int) -> None:
        head = PolicyHead(embed_dim=embed_dim)
        A = 8  # 8 candidate actions
        rule_embeds  = torch.randn(A, embed_dim)
        eclass_embeds = torch.randn(A, embed_dim)
        mask = torch.ones(A, dtype=torch.bool)

        pi, logits = head(rule_embeds, eclass_embeds, mask)
        assert pi.shape     == (A,), f"pi shape: {pi.shape}"
        assert logits.shape == (A,), f"logits shape: {logits.shape}"

    def test_sums_to_one(self, embed_dim: int) -> None:
        head = PolicyHead(embed_dim=embed_dim)
        A = 6
        mask = torch.ones(A, dtype=torch.bool)
        pi, _ = head(torch.randn(A, embed_dim), torch.randn(A, embed_dim), mask)
        assert abs(pi.sum().item() - 1.0) < 1e-5, f"pi does not sum to 1: {pi.sum().item()}"

    def test_mask_zeroes_illegal(self, embed_dim: int) -> None:
        head = PolicyHead(embed_dim=embed_dim)
        A = 5
        mask = torch.tensor([True, True, False, False, True])
        pi, _ = head(torch.randn(A, embed_dim), torch.randn(A, embed_dim), mask)
        # Illegal actions must have zero probability
        assert pi[2].item() == pytest.approx(0.0, abs=1e-6)
        assert pi[3].item() == pytest.approx(0.0, abs=1e-6)
        # Legal actions must share all probability mass
        assert abs(pi[[0, 1, 4]].sum().item() - 1.0) < 1e-5

    def test_empty_actions(self, embed_dim: int) -> None:
        head = PolicyHead(embed_dim=embed_dim)
        pi, logits = head(
            torch.empty(0, embed_dim),
            torch.empty(0, embed_dim),
            torch.empty(0, dtype=torch.bool),
        )
        assert pi.shape == (0,)


class TestValueHead:
    def test_output_shape(self, embed_dim: int) -> None:
        head = ValueHead(embed_dim=embed_dim)
        h_graph = torch.randn(4, embed_dim)
        v = head(h_graph)
        assert v.shape == (4, 1), f"value shape: {v.shape}"

    def test_output_in_range(self, embed_dim: int) -> None:
        head = ValueHead(embed_dim=embed_dim)
        h_graph = torch.randn(100, embed_dim) * 100  # large inputs
        v = head(h_graph)
        # tanh guarantees output in (-1, 1)
        assert (v >= -1.0 - 1e-5).all() and (v <= 1.0 + 1e-5).all(), \
            "Value head output outside [-1, 1]"


class TestRuleEmbeddingTable:
    def test_lookup_shapes(self, embed_dim: int, num_rules: int) -> None:
        table = RuleEmbeddingTable(num_rules=num_rules, embed_dim=embed_dim)
        idx = torch.tensor([0, 3, 7], dtype=torch.long)
        out = table(idx)
        assert out.shape == (3, embed_dim), f"embedding shape: {out.shape}"

    def test_expand(self, embed_dim: int) -> None:
        table = RuleEmbeddingTable(num_rules=8, embed_dim=embed_dim)
        table.expand(16)
        assert table.table.num_embeddings == 16


class TestAlphaRewriteNet:
    def test_forward_shapes(
        self, dummy_heterodata: HeteroData, embed_dim: int, num_rules: int
    ) -> None:
        net = AlphaRewriteNet(
            embed_dim=embed_dim,
            num_rules=num_rules,
            opcode_vocab_size=10,
        )
        A = 4
        action_rule_ids   = torch.randint(0, num_rules, (A,))
        action_eclass_ids = torch.randint(0, 3, (A,))  # 3 eclasses in dummy graph
        legal_mask        = torch.ones(A, dtype=torch.bool)

        pi, v = net(dummy_heterodata, action_rule_ids, action_eclass_ids, legal_mask)

        assert pi.shape == (A,),  f"pi shape: {pi.shape}"
        assert v.shape  == (1,),  f"v shape:  {v.shape}"
        assert abs(pi.sum().item() - 1.0) < 1e-5
        assert -1.0 - 1e-5 <= v.item() <= 1.0 + 1e-5

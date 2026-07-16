"""
tests/test_egraph_encoder.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Smoke tests for Module 1: EGraphEncoder.

Tests
-----
* test_encoder_output_shapes : checks that EGraphEncoder returns the 3
  expected tensors with the correct shapes on the dummy 3-node graph.
* test_encoder_no_children   : verifies the encoder handles a graph with
  no child edges without raising.
* test_jacobi_layer_shapes   : checks JacobiConvLayer forward shapes.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from egraph_encoder.encoder import EGraphEncoder
from egraph_encoder.gnn_layers import JacobiConvLayer


class TestEGraphEncoder:
    """Shape checks on EGraphEncoder.forward()."""

    def test_output_shapes(self, dummy_heterodata: HeteroData, embed_dim: int) -> None:
        """Encoder must return (h_node, h_eclass, h_graph) with correct shapes."""
        encoder = EGraphEncoder(embed_dim=embed_dim, opcode_vocab_size=10)
        out = encoder(dummy_heterodata)

        N_enodes   = int(dummy_heterodata["enode"].x.numel())
        N_eclasses = int(dummy_heterodata["eclass"].x.size(0))

        assert out["h_node"].shape   == (N_enodes,   embed_dim), \
            f"h_node shape mismatch: {out['h_node'].shape}"
        assert out["h_eclass"].shape == (N_eclasses, embed_dim), \
            f"h_eclass shape mismatch: {out['h_eclass'].shape}"
        assert out["h_graph"].shape  == (1,          embed_dim), \
            f"h_graph shape mismatch: {out['h_graph'].shape}"

    def test_no_children(self, embed_dim: int) -> None:
        """Encoder must not crash when child edge index is empty."""
        data = HeteroData()
        data["enode"].x  = torch.tensor([0, 1], dtype=torch.long)
        data["eclass"].x = torch.zeros((2, embed_dim), dtype=torch.float)
        data[("enode", "child_of", "eclass")].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[("enode", "child_of", "eclass")].pos        = torch.empty((0,),   dtype=torch.long)
        data[("eclass", "has_member", "enode")].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        data["eclass"].root_idx = torch.tensor([0], dtype=torch.long)

        encoder = EGraphEncoder(embed_dim=embed_dim, opcode_vocab_size=10)
        out = encoder(data)
        assert out["h_node"].shape  == (2, embed_dim)
        assert out["h_graph"].shape == (1, embed_dim)

    def test_empty_graph(self, embed_dim: int) -> None:
        """Encoder must handle a completely empty graph (0 enodes, 0 eclasses)."""
        data = HeteroData()
        data["enode"].x  = torch.empty((0,), dtype=torch.long)
        data["eclass"].x = torch.zeros((0, embed_dim), dtype=torch.float)
        data[("enode", "child_of", "eclass")].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[("enode", "child_of", "eclass")].pos        = torch.empty((0,),   dtype=torch.long)
        data[("eclass", "has_member", "enode")].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["eclass"].root_idx = None

        encoder = EGraphEncoder(embed_dim=embed_dim, opcode_vocab_size=10)
        out = encoder(data)
        assert out["h_node"].shape   == (0, embed_dim)
        assert out["h_eclass"].shape == (0, embed_dim)
        assert out["h_graph"].shape  == (1, embed_dim)

    def test_output_is_finite(self, dummy_heterodata: HeteroData, embed_dim: int) -> None:
        """All output tensors must be finite (no NaN or Inf)."""
        encoder = EGraphEncoder(embed_dim=embed_dim, opcode_vocab_size=10)
        out = encoder(dummy_heterodata)
        for key, tensor in out.items():
            assert torch.isfinite(tensor).all(), \
                f"{key} contains non-finite values: {tensor}"


class TestJacobiConvLayer:
    """Shape checks on a single JacobiConvLayer."""

    def test_forward_shapes(self, dummy_heterodata: HeteroData, embed_dim: int) -> None:
        """JacobiConvLayer forward must return correct output shapes."""
        N_e  = int(dummy_heterodata["enode"].x.numel())
        N_ec = int(dummy_heterodata["eclass"].x.size(0))

        layer = JacobiConvLayer(embed_dim=embed_dim)
        h_node_prev   = torch.randn(N_e, embed_dim)
        h_eclass_prev = torch.randn(N_ec, embed_dim)
        op_features   = torch.randn(N_e, embed_dim)

        child_store  = dummy_heterodata[("enode", "child_of", "eclass")]
        member_store = dummy_heterodata[("eclass", "has_member", "enode")]

        h_node_new, h_eclass_new = layer(
            h_node_prev=h_node_prev,
            h_eclass_prev=h_eclass_prev,
            op_features=op_features,
            child_edge_index=child_store.edge_index,
            child_pos=child_store.pos,
            member_edge_index=member_store.edge_index,
        )
        assert h_node_new.shape   == (N_e,  embed_dim)
        assert h_eclass_new.shape == (N_ec, embed_dim)

"""
egraph_encoder — Heterogeneous GNN encoder over the e-graph.

Exposes:
    EGraphEncoder      : nn.Module, the main GNN encoder
    EGraphHeteroData   : builder that converts a Learn-R EGraph → HeteroData
    JacobiConvLayer    : a single heterogeneous Jacobi message-passing layer

Design reference: "Figure 6.3: RL Encoder Input/Output Structure"
"""

from egraph_encoder.encoder import EGraphEncoder
from egraph_encoder.hetero_data import EGraphHeteroData
from egraph_encoder.gnn_layers import JacobiConvLayer

__all__ = [
    "EGraphEncoder",
    "EGraphHeteroData",
    "JacobiConvLayer",
]

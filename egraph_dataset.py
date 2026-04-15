"""Utilities to convert EGraph structures into PyG HeteroData."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from egraph_bridge.simple_egraph import EGraph, ENode
from rl_encoder.opcode_vocab import get_opcode_idx


def _active_eclass_ids(egraph: EGraph) -> List[int]:
    """Return canonical e-class IDs that are currently active roots."""
    active: List[int] = []
    for eclass_id in egraph.eclasses:
        canonical_id = egraph._find(eclass_id)
        if canonical_id == eclass_id:
            active.append(eclass_id)
    active.sort()
    return active


def egraph_to_heterodata(
    egraph: EGraph,
    root_eclass_id: Optional[int] = None,
) -> HeteroData:
    """Convert an EGraph into a heterogeneous graph for neural encoding."""
    data = HeteroData()

    active_eclasses = _active_eclass_ids(egraph)
    eclass_idx_map: Dict[int, int] = {eid: i for i, eid in enumerate(active_eclasses)}

    enodes: List[Tuple[int, ENode]] = []
    enode_idx_map: Dict[Tuple[int, int], int] = {}
    for eid in active_eclasses:
        eclass = egraph.eclasses[eid]
        for local_idx, enode in enumerate(eclass.nodes):
            idx = len(enodes)
            enodes.append((eid, enode))
            enode_idx_map[(eid, local_idx)] = idx

    num_eclasses = len(active_eclasses)
    num_enodes = len(enodes)

    x_enode = torch.empty((num_enodes,), dtype=torch.long)
    for i, (_, enode) in enumerate(enodes):
        x_enode[i] = get_opcode_idx(enode.op)
    data["enode"].x = x_enode

    data["eclass"].x = torch.zeros((num_eclasses, 1), dtype=torch.float)

    child_src: List[int] = []
    child_dst: List[int] = []
    child_pos: List[int] = []

    for node_idx, (_, enode) in enumerate(enodes):
        for pos, child in enumerate(enode.children):
            if isinstance(child, int):
                child_canonical = egraph._find(child)
                if child_canonical in eclass_idx_map:
                    child_src.append(node_idx)
                    child_dst.append(eclass_idx_map[child_canonical])
                    child_pos.append(pos)

    if child_src:
        child_edge_index = torch.tensor([child_src, child_dst], dtype=torch.long)
        child_pos_tensor = torch.tensor(child_pos, dtype=torch.long)
    else:
        child_edge_index = torch.empty((2, 0), dtype=torch.long)
        child_pos_tensor = torch.empty((0,), dtype=torch.long)

    data[("enode", "child_of", "eclass")].edge_index = child_edge_index
    data[("enode", "child_of", "eclass")].pos = child_pos_tensor

    member_src: List[int] = []
    member_dst: List[int] = []
    for eid in active_eclasses:
        eclass = egraph.eclasses[eid]
        src_idx = eclass_idx_map[eid]
        for local_idx, _ in enumerate(eclass.nodes):
            node_idx = enode_idx_map[(eid, local_idx)]
            member_src.append(src_idx)
            member_dst.append(node_idx)

    if member_src:
        member_edge_index = torch.tensor([member_src, member_dst], dtype=torch.long)
    else:
        member_edge_index = torch.empty((2, 0), dtype=torch.long)

    data[("eclass", "has_member", "enode")].edge_index = member_edge_index

    if root_eclass_id is not None:
        root_canonical = egraph._find(root_eclass_id)
        if root_canonical not in eclass_idx_map:
            raise KeyError(
                f"Root eclass id {root_eclass_id} resolves to inactive class {root_canonical}"
            )
        data["eclass"].root_idx = torch.tensor([eclass_idx_map[root_canonical]], dtype=torch.long)
    else:
        data["eclass"].root_idx = None

    return data

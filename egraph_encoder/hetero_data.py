"""
egraph_encoder/hetero_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Converts a **Learn-R EGraph** object (from ``egraph_bridge.simple_egraph``)
into the 4-part PyTorch Geometric ``HeteroData`` structure consumed by
:class:`EGraphEncoder`.

Graph schema
------------
Node types:
  * ``"enode"``  — operation nodes.  Feature: integer opcode index, shape ``[N_enodes]``.
  * ``"eclass"`` — equivalence-class nodes.  Feature: learnable init vector
    placeholder of shape ``[N_eclasses, init_dim]``.

Edge types:
  * ``("enode",  "child_of",   "eclass")`` — enode → e-class it uses as child operand.
    Carries ``pos`` attribute (child position / slot index).
  * ``("eclass", "has_member", "enode")``  — e-class → enodes that are members of it.

Usage
-----
>>> builder = EGraphHeteroData(init_dim=128)
>>> hetero = builder.build(egraph, root_eclass_id=0)
>>> hetero["enode"].x.shape         # [N_enodes]
>>> hetero["eclass"].x.shape        # [N_eclasses, 128]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# Attempt to import the real Learn-R EGraph; fall back to a stub so the
# module stays importable in isolated environments (e.g. unit tests).
# ---------------------------------------------------------------------------
try:
    from egraph_bridge.simple_egraph import EGraph, ENode  # type: ignore[import]
except ImportError:  # pragma: no cover
    EGraph = None  # type: ignore[assignment,misc]
    ENode = None   # type: ignore[assignment,misc]

try:
    from rl_encoder.opcode_vocab import get_opcode_idx  # type: ignore[import]
except ImportError:  # pragma: no cover
    def get_opcode_idx(op: str) -> int:  # type: ignore[misc]
        """Fallback: always return 0 (unknown opcode)."""
        return 0


class EGraphHeteroData:
    """Build a :class:`~torch_geometric.data.HeteroData` from a Learn-R EGraph.

    Parameters
    ----------
    init_dim : int
        Dimensionality of the learnable placeholder feature vector stored in
        ``data["eclass"].x``.  Typically equal to ``EGraphEncoder.embed_dim``.
    """

    def __init__(self, init_dim: int = 128) -> None:
        self.init_dim = init_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        egraph: "EGraph",
        root_eclass_id: Optional[int] = None,
    ) -> HeteroData:
        """Convert *egraph* into a :class:`HeteroData` ready for
        :class:`EGraphEncoder`.

        Parameters
        ----------
        egraph : EGraph
            A Learn-R e-graph produced by the pipeline.
        root_eclass_id : int, optional
            The canonical e-class ID of the program root, used to identify
            the root during graph-level readout.  When ``None``, attention
            pooling is used as a fallback.

        Returns
        -------
        HeteroData
            ``data["enode"].x``     : LongTensor  [N_enodes]
            ``data["eclass"].x``    : FloatTensor [N_eclasses, init_dim]
            child edges  ``("enode","child_of","eclass")``  — with ``.pos``
            member edges ``("eclass","has_member","enode")``
            ``data["eclass"].root_idx`` : LongTensor [1] or None
        """
        # TODO: wire to real EGraph once the API is stable.
        data = HeteroData()

        active_eclasses, eclass_idx_map = self._collect_eclasses(egraph)
        enodes, enode_idx_map = self._collect_enodes(egraph, active_eclasses)

        # --- node features ------------------------------------------------
        data["enode"].x = self._build_enode_features(enodes)
        data["eclass"].x = self._build_eclass_features(len(active_eclasses))

        # --- edge indices -------------------------------------------------
        child_ei, child_pos = self._build_child_edges(
            egraph, enodes, eclass_idx_map
        )
        data[("enode", "child_of", "eclass")].edge_index = child_ei
        data[("enode", "child_of", "eclass")].pos = child_pos

        member_ei = self._build_member_edges(
            egraph, active_eclasses, eclass_idx_map, enode_idx_map
        )
        data[("eclass", "has_member", "enode")].edge_index = member_ei

        # --- optional root marker -----------------------------------------
        if root_eclass_id is not None:
            # TODO: handle union-find canonical resolution via egraph._find()
            canonical = egraph._find(root_eclass_id)
            if canonical not in eclass_idx_map:
                raise KeyError(
                    f"root_eclass_id {root_eclass_id} -> canonical {canonical} "
                    "not found in active e-classes"
                )
            data["eclass"].root_idx = torch.tensor(
                [eclass_idx_map[canonical]], dtype=torch.long
            )
        else:
            data["eclass"].root_idx = None

        return data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_eclasses(egraph: "EGraph") -> Tuple[List[int], Dict[int, int]]:
        """Return sorted active e-class IDs and a local-index map.

        Returns
        -------
        active : List[int]
            Canonical e-class IDs that are currently live roots.
        eclass_idx_map : Dict[int, int]
            Maps canonical eclass_id -> dense row index in [0, N_eclasses).
        """
        # TODO: replace with egraph canonical-root enumeration once API stable.
        active: List[int] = []
        for eclass_id in egraph.eclasses:
            if egraph._find(eclass_id) == eclass_id:
                active.append(eclass_id)
        active.sort()
        eclass_idx_map = {eid: i for i, eid in enumerate(active)}
        return active, eclass_idx_map

    @staticmethod
    def _collect_enodes(
        egraph: "EGraph",
        active_eclasses: List[int],
    ) -> Tuple[List[Tuple[int, "ENode"]], Dict[Tuple[int, int], int]]:
        """Flatten all enodes across active e-classes into a dense list.

        Returns
        -------
        enodes : List[(eclass_id, ENode)]
            Flat ordered list of (owning_eclass_id, enode) pairs.
        enode_idx_map : Dict[(eclass_id, local_slot), int]
            Maps (eclass_id, local_slot_index) -> dense row index.
        """
        # TODO: iterate egraph.eclasses[eid].nodes once EGraph API confirmed.
        enodes: List[Tuple[int, "ENode"]] = []
        enode_idx_map: Dict[Tuple[int, int], int] = {}
        for eid in active_eclasses:
            eclass = egraph.eclasses[eid]
            for slot, enode in enumerate(eclass.nodes):
                idx = len(enodes)
                enodes.append((eid, enode))
                enode_idx_map[(eid, slot)] = idx
        return enodes, enode_idx_map

    @staticmethod
    def _build_enode_features(enodes: List[Tuple[int, "ENode"]]) -> Tensor:
        """Return opcode-index tensor for all enodes.

        Returns
        -------
        Tensor  shape [N_enodes], dtype=long
            Integer opcode index per enode (from the shared opcode vocabulary).
        """
        # TODO: extend to richer per-enode features (constant values, flags).
        x = torch.empty((len(enodes),), dtype=torch.long)
        for i, (_, enode) in enumerate(enodes):
            x[i] = get_opcode_idx(enode.op)
        return x

    def _build_eclass_features(self, num_eclasses: int) -> Tensor:
        """Return zero-initialised eclass feature matrix.

        The encoder projects this through a learnable embedding; zeros are
        a valid neutral initialisation.

        Returns
        -------
        Tensor  shape [N_eclasses, init_dim], dtype=float
        """
        # TODO: optionally seed with PHI/Mem metadata from the EqSat graph.
        return torch.zeros((num_eclasses, self.init_dim), dtype=torch.float)

    @staticmethod
    def _build_child_edges(
        egraph: "EGraph",
        enodes: List[Tuple[int, "ENode"]],
        eclass_idx_map: Dict[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """Build child edges (enode -> e-class).

        Returns
        -------
        edge_index : LongTensor  shape [2, E_child]
            Row 0 = source enode index, Row 1 = target eclass index.
        pos : LongTensor  shape [E_child]
            Child-slot position (0-based operand index).
        """
        # TODO: handle implicit literal children that have no e-class.
        src, dst, pos_list = [], [], []
        for node_idx, (_, enode) in enumerate(enodes):
            for slot, child in enumerate(enode.children):
                if isinstance(child, int):
                    canonical = egraph._find(child)
                    if canonical in eclass_idx_map:
                        src.append(node_idx)
                        dst.append(eclass_idx_map[canonical])
                        pos_list.append(slot)
        if src:
            return (
                torch.tensor([src, dst], dtype=torch.long),
                torch.tensor(pos_list, dtype=torch.long),
            )
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )

    @staticmethod
    def _build_member_edges(
        egraph: "EGraph",
        active_eclasses: List[int],
        eclass_idx_map: Dict[int, int],
        enode_idx_map: Dict[Tuple[int, int], int],
    ) -> Tensor:
        """Build membership edges (e-class -> enode).

        Returns
        -------
        edge_index : LongTensor  shape [2, E_member]
            Row 0 = source eclass index, Row 1 = target enode index.
        """
        src, dst = [], []
        for eid in active_eclasses:
            eclass = egraph.eclasses[eid]
            src_idx = eclass_idx_map[eid]
            for slot, _ in enumerate(eclass.nodes):
                node_idx = enode_idx_map[(eid, slot)]
                src.append(src_idx)
                dst.append(node_idx)
        if src:
            return torch.tensor([src, dst], dtype=torch.long)
        return torch.empty((2, 0), dtype=torch.long)

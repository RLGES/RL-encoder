"""
tests/conftest.py
~~~~~~~~~~~~~~~~~~
Shared pytest fixtures for RL-encoder smoke tests.

Provides a minimal 3-node dummy e-graph that exercises the full
heterogeneous graph structure without requiring Learn-R at all.
Tests that need torch_geometric are automatically skipped when
the library is not installed.

Dummy graph structure
----------------------
Term: (x + 0)

  EClass 0: {enode 0: ADD(ec1, ec2)}   ← root
  EClass 1: {enode 1: REG("x")}
  EClass 2: {enode 2: CONST(0)}

  Child edges:  0→1 (slot 0),  0→2 (slot 1)
  Member edges: eclass0→enode0, eclass1→enode1, eclass2→enode2
"""

from __future__ import annotations

import sys
import types
from pathlib import Path



# Add RL-encoder and Learn-R to sys.path so modules resolve correctly
_TESTS_DIR = Path(__file__).parent.resolve()
_RL_ENCODER_DIR = _TESTS_DIR.parent
_LEARNR_DIR = _RL_ENCODER_DIR.parent / "Learn-R"

for p in [str(_RL_ENCODER_DIR), str(_LEARNR_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Register rl_encoder alias so the root __init__.py works
if "rl_encoder" not in sys.modules:
    rl_pkg = types.ModuleType("rl_encoder")
    rl_pkg.__path__ = [str(_RL_ENCODER_DIR)]           # type: ignore[assignment]
    rl_pkg.__package__ = "rl_encoder"
    rl_pkg.__spec__ = None                   # type: ignore[assignment]
    sys.modules["rl_encoder"] = rl_pkg

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures that do NOT require torch_geometric
# ---------------------------------------------------------------------------

@pytest.fixture
def embed_dim() -> int:
    return 128


@pytest.fixture
def num_rules() -> int:
    return 16


# ---------------------------------------------------------------------------
# Fixtures that REQUIRE torch_geometric (auto-skip when not installed)
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_heterodata():
    """Return a 3-enode / 3-eclass HeteroData for smoke tests.

    Automatically skips the test if torch_geometric is not installed.

    Tensor shapes:
      enode.x                  : [3]       (opcode indices 0, 1, 2)
      eclass.x                 : [3, 128]  (zero-init)
      (enode,child_of,eclass)  : edge_index [2, 2], pos [2]
      (eclass,has_member,enode): edge_index [2, 3]
      eclass.root_idx          : [1]       (= 0)
    """
    torch_geometric = pytest.importorskip(
        "torch_geometric", reason="torch_geometric not installed"
    )
    from torch_geometric.data import HeteroData

    data = HeteroData()

    # Enode features: opcode indices for ADD, REG, CONST
    data["enode"].x = torch.tensor([0, 1, 2], dtype=torch.long)

    # Eclass features: zeros (128-dim)
    data["eclass"].x = torch.zeros((3, 128), dtype=torch.float)

    # Child edges: enode 0 → eclass 1 (slot 0), enode 0 → eclass 2 (slot 1)
    data[("enode", "child_of", "eclass")].edge_index = torch.tensor(
        [[0, 0], [1, 2]], dtype=torch.long
    )
    data[("enode", "child_of", "eclass")].pos = torch.tensor([0, 1], dtype=torch.long)

    # Member edges: eclass i → enode i (1-1 in this dummy)
    data[("eclass", "has_member", "enode")].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )

    # Root marker
    data["eclass"].root_idx = torch.tensor([0], dtype=torch.long)

    return data

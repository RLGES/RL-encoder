# RL Encoder

This package implements an e-graph to tensor encoder for an AlphaZero-style RL agent that guides equality saturation.

## Overview

The module provides:

- Opcode and rewrite-rule vocabularies.
- E-graph to `torch_geometric.data.HeteroData` conversion.
- A Jacobi-style heterogeneous GNN encoder.
- Policy and value heads for valid rewrite-action scoring and graph value prediction.
- Utility helpers for batching e-graphs, reward computation, and feature-based rule embedding initialization.

## Package Layout

- `opcode_vocab.py`: opcode/rule vocabulary registration and lookup.
- `egraph_dataset.py`: conversion from `EGraph` to `HeteroData`.
- `hetero_encoder.py`: `EGraphEncoder` with layer-wise Jacobi updates.
- `policy_value_heads.py`: `PolicyHead`, `ValueHead`, and `PolicyValueNetwork`.
- `encoder_utils.py`: batching, reward, and rule initialization utilities.
- `__init__.py`: public exports.

## Key Design Constraints

- Uses existing core types from the project (`EGraph`, `ENode`, `EClass`, `RewriteRule`, `Instruction`).
- Separate embeddings for opcode entities and rewrite-rule entities.
- Policy softmax is applied only over valid actions.
- E-class resolution uses `egraph._find(...)` canonicalization.
- Supports PyG `HeteroData` batching for multi-graph training.

## Expected Dependencies

The module expects:

- `torch`
- `torch_geometric`

If imports fail, install the dependencies in the active environment.

## Public API

Exported symbols:

- `EGraphEncoder`
- `PolicyValueNetwork`
- `PolicyHead`
- `ValueHead`
- `egraph_to_heterodata`
- `batch_egraphs`
- `register_rule`
- `compute_reward`

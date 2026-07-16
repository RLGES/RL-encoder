# RL-encoder

> **Module 2 of EqSat** — RL-guided equality saturation engine.
> Consumes Learn-R's verified Active Rule Set + E-Graph output and runs
> an AlphaZero-style policy/value network to search for optimal rewrites.

---

## Architecture Overview

```
Learn-R (upstream, read-only)
  ├── LLM Rule Generator   → raw candidate rules
  ├── Z3 SMT Verifier      → verified rules
  ├── Rule DB              → Active Rule Set (|A| rules)
  └── E-Graph Engine       → EGraph object (eclasses + enodes)
           │
           ▼
RL-encoder (this module)
  ├── egraph_encoder/      MODULE 1  — Heterogeneous GNN encoder
  ├── model/               MODULE 2  — Policy + Value heads
  ├── data_generation/     MODULE 3  — Self-play data pipeline
  ├── training/            MODULE 4  — AlphaZero outer loop
  └── inference/           MODULE 5  — Inference optimizer + CLI
```

---

## Module Map

| Directory | Purpose | Key Class |
|---|---|---|
| `egraph_encoder/` | Jacobi-style hetero GNN → node / eclass / graph embeddings | `EGraphEncoder` |
| `model/` | Policy head (masked softmax), value head (tanh), rule table | `AlphaRewriteNet` |
| `data_generation/` | EGraphServer, MCTS (PUCT), self-play loop, replay buffer | `play_game()` |
| `training/` | AlphaZero outer loop, loss, checkpoints, config | `outer_loop()` |
| `inference/` | Greedy inference optimizer + CLI | `AlphaRewriteOptimizer` |

---

## Framework Choice: PyTorch Geometric (PyG)

We use **PyTorch Geometric** (`torch_geometric`) for the heterogeneous GNN
because:

1. **Native HeteroData support** — PyG's `HeteroData` maps directly to our
   4-component graph schema (enode features, eclass features, child edges,
   member edges) with zero boilerplate.
2. **Batch collation** — `Batch.from_data_list()` handles mini-batching of
   variable-size e-graphs automatically.
3. **`scatter` / `softmax` primitives** — GPU-optimised segment operations
   are used throughout the Jacobi message-passing layers.
4. **Community** — larger ecosystem of GNN primitives to extend later.

DGL was the main alternative; PyG was chosen for its cleaner heterogeneous
graph API.

---

## Module 1: EGraphEncoder

**Files**: `egraph_encoder/hetero_data.py`, `gnn_layers.py`, `encoder.py`

### Input Schema (HeteroData)

| Component | Shape | Semantics |
|---|---|---|
| `data["enode"].x` | `[N_enodes]` long | Integer opcode index per enode |
| `data["eclass"].x` | `[N_eclasses, init_dim]` float | Zero-init eclass feature placeholders |
| `("enode","child_of","eclass").edge_index` | `[2, E_child]` long | enode→eclass operand edges |
| `("enode","child_of","eclass").pos` | `[E_child]` long | Child slot index (0-based) |
| `("eclass","has_member","enode").edge_index` | `[2, E_member]` long | eclass→enode membership edges |

### Output

| Tensor | Shape | Semantics |
|---|---|---|
| `h_node` | `[N_enodes, embed_dim]` | Contextualized per-operation representations |
| `h_eclass` | `[N_eclasses, embed_dim]` | Aggregated per-equivalence-class representations |
| `h_graph` | `[B, embed_dim]` | Attention-pooled global graph embedding |

### Jacobi Layers

Each `JacobiConvLayer` performs one iteration:
1. **enode update**: aggregate child e-class messages (position-embedded) → MLP
2. **eclass update**: cross-attend over member enodes → project back to embed_dim

Both updates read from the **previous iteration's** embeddings (Jacobi, not Gauss-Seidel).

---

## Module 2: Policy & Value Heads

**Files**: `model/rule_embeddings.py`, `model/policy_head.py`, `model/value_head.py`, `model/alpha_rewrite_net.py`

```
Action = (rule_idx, eclass_idx)

PolicyHead forward:
  rule_embed = RuleEmbeddingTable[rule_idx]    [|A|, D]
  H_i        = h_eclass[eclass_idx]            [|A|, D]
  x          = concat([rule_embed, H_i])       [|A|, 2D]
  logits     = MLP_2layer(x)                   [|A|]
  masked     = logits.masked_fill(~legal, -∞)
  pi         = softmax(masked)                 ∈ Δ^|A|

ValueHead forward:
  v = tanh(MLP_2layer(h_G))                   ∈ (-1, +1)
```

---

## Module 3: Data Generation / Self-Play

**Files**: `data_generation/egraph_server.py`, `mcts.py`, `self_play.py`, `replay_buffer.py`, `metrics.py`

```python
def play_game(egraph_server, mcts_server, inference_server, T0, budget):
    G = egraph_server.egraph(T0)
    B = []
    while not terminal(G) and budget > 0:
        p_hat, v_hat = inference_server.predict(G)
        pi = mcts_server.puct(G, p_hat, v_hat)
        B.append((G, pi, None))
        G = egraph_server.apply(G, sample(pi))
        budget -= 1
    z = compute_final_value(G)
    return [(g, pi, z) for (g, pi, _) in B]
```

### Replay Buffer JSON Schema

```json
{
  "trajectory_id": "uuid",
  "step_in_game": 3,
  "state_G": {
    "description": "x * 2 + 0",
    "num_eclasses": 4,
    "num_enodes": 5
  },
  "mcts_policy_pi": { "rule_add_zero__ec1": 0.72, "rule_mul_one__ec2": 0.28 },
  "target_value_z": 0.35,
  "cost_metadata": {
    "initial_cost": 8.0,
    "final_cost": 5.0,
    "cost_metric": "instruction_count"
  }
}
```

---

## Module 4: Training

**Files**: `training/config.py`, `training/loss.py`, `training/checkpoint.py`, `training/train_loop.py`

```
Loss = MSE(v, z) + CrossEntropy(π_pred, π_mcts) + λ‖θ‖²

Outer loop:
  θ₀ = init(); publish(θ₀)
  for k in 0..K-1:
    D_k = run_self_play(N_games, θ_k)
    θ_{k+1} = train(D_k, T_steps)
    publish(θ_{k+1})
```

---

## Module 5: Inference

**Files**: `inference/optimizer.py`, `inference/cli.py`

```python
opt = AlphaRewriteOptimizer("checkpoints/theta_k0019.pt")
result = opt.optimize("x * 2 + 0", budget=50)
print(result.optimized_term)   # "x * 2"
```

CLI:
```bash
python -m inference.cli \
  --checkpoint checkpoints/theta_k0019.pt \
  --input test_code.txt \
  --budget 50
```

---

## Smoke Test: End-to-End with a 3-Node Dummy E-Graph

```bash
cd RL-encoder
pip install torch torch-geometric pytest

python - <<'EOF'
import torch
from torch_geometric.data import HeteroData
from egraph_encoder.encoder import EGraphEncoder
from model.alpha_rewrite_net import AlphaRewriteNet

# Build dummy 3-node e-graph: (x + 0)  — 3 eclasses, 3 enodes
data = HeteroData()
data["enode"].x  = torch.tensor([0, 1, 2], dtype=torch.long)   # ADD, REG, CONST
data["eclass"].x = torch.zeros((3, 128), dtype=torch.float)
data[("enode","child_of","eclass")].edge_index = torch.tensor([[0,0],[1,2]], dtype=torch.long)
data[("enode","child_of","eclass")].pos        = torch.tensor([0, 1], dtype=torch.long)
data[("eclass","has_member","enode")].edge_index = torch.tensor([[0,1,2],[0,1,2]], dtype=torch.long)
data["eclass"].root_idx = torch.tensor([0], dtype=torch.long)

net = AlphaRewriteNet(embed_dim=128, num_rules=16, opcode_vocab_size=10)
pi, v = net(
    hetero_data=data,
    action_rule_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    action_eclass_ids=torch.tensor([0, 0, 1, 2], dtype=torch.long),
    legal_action_mask=torch.ones(4, dtype=torch.bool),
)
print(f"pi={pi.tolist()}  (sums to {pi.sum():.6f})")
print(f"v={v.item():.4f}  (in (-1, +1))")
assert abs(pi.sum() - 1.0) < 1e-5
assert -1.0 < v.item() < 1.0
print("Smoke test PASSED.")
EOF
```

Run all smoke tests:

```bash
cd RL-encoder
pytest tests/ -v
```

---

## Plugging into Learn-R

```python
# In your Learn-R pipeline:
from egraph_bridge.simple_egraph import EGraph
from egraph_encoder.hetero_data import EGraphHeteroData

egraph: EGraph = ...  # produced by Learn-R
builder = EGraphHeteroData(init_dim=128)
hetero  = builder.build(egraph, root_eclass_id=0)

# Then feed to AlphaRewriteNet:
from model.alpha_rewrite_net import AlphaRewriteNet
net = AlphaRewriteNet(...)
pi, v = net(hetero, action_rule_ids, action_eclass_ids, legal_mask)
```

---

## File Tree

```
RL-encoder/
├── README.md                        ← this file
├── __init__.py                      ← existing public API
├── egraph_dataset.py                ← existing (Learn-R bridge)
├── encoder_utils.py                 ← existing (batch helpers)
├── hetero_encoder.py                ← existing EGraphEncoder (legacy)
├── opcode_vocab.py                  ← existing (shared vocab)
├── policy_value_heads.py            ← existing (legacy heads)
│
├── egraph_encoder/                  MODULE 1 — Heterogeneous GNN
│   ├── __init__.py
│   ├── hetero_data.py               EGraphHeteroData builder
│   ├── gnn_layers.py                JacobiConvLayer
│   └── encoder.py                   EGraphEncoder
│
├── model/                           MODULE 2 — Policy + Value heads
│   ├── __init__.py
│   ├── rule_embeddings.py           RuleEmbeddingTable
│   ├── policy_head.py               PolicyHead
│   ├── value_head.py                ValueHead
│   └── alpha_rewrite_net.py         AlphaRewriteNet (top-level)
│
├── data_generation/                 MODULE 3 — Self-play pipeline
│   ├── __init__.py
│   ├── egraph_server.py             EGraphServer adapter
│   ├── mcts.py                      MCTSNode + MCTSServer (PUCT)
│   ├── self_play.py                 play_game() + InferenceServer
│   ├── replay_buffer.py             ReplayBuffer (circular + JSONL)
│   └── metrics.py                   GameMetrics logger
│
├── training/                        MODULE 4 — Training loop
│   ├── __init__.py
│   ├── config.py                    TrainingConfig dataclass
│   ├── loss.py                      alpha_zero_loss()
│   ├── checkpoint.py                save / load / publish
│   └── train_loop.py                outer_loop()
│
├── inference/                       MODULE 5 — Inference
│   ├── __init__.py
│   ├── optimizer.py                 AlphaRewriteOptimizer
│   └── cli.py                       python -m inference.cli
│
└── tests/
    ├── __init__.py
    ├── conftest.py                  Shared fixtures (dummy 3-node graph)
    ├── test_egraph_encoder.py       Module 1 smoke tests
    ├── test_model.py                Module 2 smoke tests
    ├── test_data_generation.py      Module 3 smoke tests
    ├── test_training.py             Module 4 smoke tests
    └── test_inference.py            Module 5 smoke tests
```

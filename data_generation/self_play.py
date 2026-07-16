"""
data_generation/self_play.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core self-play loop implementing the "PLAY GAME OPTIMIZATION ALGORITHM".

Design reference: "PHASE 2: RL OPTIMIZATION ENGINE" slide

Algorithm (single game)
-----------------------
  G = EGraphServer.egraph(T0)
  B = []
  while not terminal(G) and budget > 0:
      p̂, v̂ = InferenceServer.predict(G)
      pi     = MCTSServer.puct(G, p̂, v̂)
      B.append((G, pi, None))          # value placeholder
      action = sample(pi)
      G      = EGraphServer.apply(G, action)
      budget -= 1
  z = compute_final_value(G)            # cost-based terminal reward
  B = [(G, pi, z) for (G, pi, _) in B]
  return B

The outer loop (``run_self_play``) repeats this N_games times, collecting
all tuples into the :class:`~data_generation.replay_buffer.ReplayBuffer`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from data_generation.egraph_server import Action, EGraphServer
from data_generation.mcts import MCTSServer
from data_generation.metrics import GameMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EGraph   = Any  # Learn-R EGraph (avoid hard import at module level)
Policy   = Dict[Action, float]   # pi: action -> probability (sums to 1)
Sample   = Tuple[EGraph, Policy, Optional[float]]  # (G, pi, z)


# ---------------------------------------------------------------------------
# InferenceServer (thin wrapper around AlphaRewriteNet)
# ---------------------------------------------------------------------------

class InferenceServer:
    """Wraps :class:`~model.alpha_rewrite_net.AlphaRewriteNet` for inference.

    The inference server maintains a reference to the *current published
    checkpoint* weights.  The training loop calls :meth:`load` whenever new
    weights are available.

    Parameters
    ----------
    net : AlphaRewriteNet
        The network to wrap.
    egraph_server : EGraphServer
        Used to enumerate legal actions (needed to build the action tensors).
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).

    Usage
    -----
    >>> server = InferenceServer(net, egraph_server)
    >>> prior, value = server.predict(egraph_state)
    """

    def __init__(
        self,
        net: Any,            # AlphaRewriteNet
        egraph_server: EGraphServer,
        device: str = "cpu",
    ) -> None:
        self.net = net
        self.egraph_server = egraph_server
        self.device = device

    def predict(self, egraph: EGraph) -> Tuple[Policy, float]:
        """Run inference on *egraph* and return (prior, value).

        Parameters
        ----------
        egraph : EGraph

        Returns
        -------
        prior : Dict[Action, float]
            Prior policy distribution over legal actions (from policy head).
        value : float
            State value estimate from value head, in ``(-1, +1)``.
        """
        import torch
        from egraph_encoder.hetero_data import EGraphHeteroData

        legal_actions = self.egraph_server.legal_actions(egraph)
        if not legal_actions:
            return {}, 0.0

        # Build HeteroData.
        builder = EGraphHeteroData(init_dim=self.net.encoder.embed_dim)
        try:
            hetero = builder.build(egraph)
        except Exception as exc:
            logger.warning("HeteroData build failed: %s. Returning uniform prior.", exc)
            uniform = 1.0 / len(legal_actions)
            return {a: uniform for a in legal_actions}, 0.0

        # Build action tensors.
        rule_ids   = torch.tensor([a.rule_id   for a in legal_actions], dtype=torch.long)
        eclass_ids = torch.tensor([a.eclass_id for a in legal_actions], dtype=torch.long)
        legal_mask = torch.ones(len(legal_actions), dtype=torch.bool)

        with torch.no_grad():
            pi_tensor, v_tensor = self.net.predict(
                hetero_data=hetero,
                action_rule_ids=rule_ids,
                action_eclass_ids=eclass_ids,
                legal_action_mask=legal_mask,
            )

        pi_vals = pi_tensor.cpu().tolist()
        value   = float(v_tensor.squeeze().item())
        prior   = {a: float(p) for a, p in zip(legal_actions, pi_vals)}
        return prior, value

    def load(self, state_dict: Any) -> None:
        """Replace network weights with *state_dict* (from checkpoint).

        Parameters
        ----------
        state_dict : dict
            PyTorch ``state_dict`` from :func:`~training.checkpoint.save`.
        """
        # TODO: thread-safe weight swap for concurrent self-play + training.
        self.net.load_state_dict(state_dict)
        self.net.eval()


# ---------------------------------------------------------------------------
# Core game loop
# ---------------------------------------------------------------------------

def play_game(
    egraph_server: EGraphServer,
    mcts_server: MCTSServer,
    inference_server: InferenceServer,
    initial_term: str,
    budget: int,
    temperature: float = 1.0,
) -> List[Sample]:
    """Play one self-play game and return training samples.

    Parameters
    ----------
    egraph_server : EGraphServer
    mcts_server : MCTSServer
    inference_server : InferenceServer
    initial_term : str
        Source term to optimise (e.g. ``"x * 2 + 0"``).
    budget : int
        Maximum number of rewrite steps per game.
    temperature : float
        Sampling temperature for action selection from MCTS policy.
        ``1.0`` = proportional to visit counts; ``→ 0`` = greedy.

    Returns
    -------
    samples : List[(EGraph, Policy, float)]
        Training tuples where the third element ``z`` is the final reward,
        retroactively filled in after the game ends.
    """
    trajectory_id = str(uuid.uuid4())
    logger.info("Game %s: starting on term %r, budget=%d", trajectory_id, initial_term, budget)

    G = egraph_server.egraph(initial_term)
    initial_cost = egraph_server.cost(G)
    buffer: List[Tuple[EGraph, Policy, None]] = []

    step = 0
    while not egraph_server.is_terminal(G) and step < budget:
        # Inference: get prior + value estimate.
        prior, value = inference_server.predict(G)

        if not prior:  # no legal actions
            break

        # MCTS: improve policy via tree search.
        pi, _ = mcts_server.puct(
            egraph=G,
            prior_policy=prior,
            value_estimate=value,
            inference_fn=inference_server.predict,
        )

        buffer.append((G, pi, None))

        # Sample action from improved policy.
        action = _sample_action(pi, temperature=temperature)
        if action is None:
            break

        G = egraph_server.apply(G, action)
        step += 1

    # Compute terminal reward z.
    final_cost = egraph_server.cost(G)
    z = _compute_final_value(initial_cost, final_cost)

    logger.info(
        "Game %s finished: steps=%d, initial_cost=%.2f, final_cost=%.2f, z=%.4f",
        trajectory_id, step, initial_cost, final_cost, z,
    )

    # Retroactively fill value targets.
    samples: List[Sample] = [(g, pi_step, z) for (g, pi_step, _) in buffer]
    return samples


def run_self_play(
    egraph_server: EGraphServer,
    mcts_server: MCTSServer,
    inference_server: InferenceServer,
    terms: List[str],
    n_games: int,
    budget: int,
    replay_buffer: Any,  # ReplayBuffer
    temperature: float = 1.0,
) -> None:
    """Run N self-play games and push samples into *replay_buffer*.

    Parameters
    ----------
    egraph_server, mcts_server, inference_server : servers
    terms : List[str]
        Pool of source terms to sample from.  Terms are drawn round-robin.
    n_games : int
        Total number of games to play.
    budget : int
        Step budget per game.
    replay_buffer : ReplayBuffer
    temperature : float
    """
    import random
    metrics = GameMetrics()

    for game_idx in range(n_games):
        term = terms[game_idx % len(terms)]
        samples = play_game(
            egraph_server=egraph_server,
            mcts_server=mcts_server,
            inference_server=inference_server,
            initial_term=term,
            budget=budget,
            temperature=temperature,
        )
        replay_buffer.extend(samples)
        metrics.record_game(len(samples))
        logger.info("Self-play round %d/%d: %d samples collected.", game_idx + 1, n_games, len(samples))

    metrics.log_summary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_action(policy: Policy, temperature: float = 1.0) -> Optional[Action]:
    """Sample an action from *policy* using the given temperature.

    Parameters
    ----------
    policy : Dict[Action, float]
        Probability distribution (sums to ~1).
    temperature : float
        ``1.0`` = proportional sampling; ``→ 0`` = argmax (greedy).

    Returns
    -------
    Action or None if policy is empty.
    """
    import random
    import math

    if not policy:
        return None

    actions = list(policy.keys())
    probs   = list(policy.values())

    if temperature < 1e-3:
        # Greedy.
        return actions[probs.index(max(probs))]

    # Temperature scaling.
    scaled = [p ** (1.0 / temperature) for p in probs]
    total  = sum(scaled) or 1.0
    scaled = [s / total for s in scaled]

    r = random.random()
    cumsum = 0.0
    for action, prob in zip(actions, scaled):
        cumsum += prob
        if r <= cumsum:
            return action
    return actions[-1]  # fallback


def _compute_final_value(initial_cost: float, final_cost: float) -> float:
    """Normalised cost-improvement reward, clipped to ``[-1, +1]``.

    Parameters
    ----------
    initial_cost : float
    final_cost   : float

    Returns
    -------
    float
        Positive = improvement; ``0`` = no change; negative = regression.
    """
    if initial_cost <= 0:
        return 0.0
    raw = (initial_cost - final_cost) / initial_cost
    return max(-1.0, min(1.0, raw))

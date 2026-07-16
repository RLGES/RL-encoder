"""
inference/optimizer.py
~~~~~~~~~~~~~~~~~~~~~~~
:class:`AlphaRewriteOptimizer` — the top-level inference entry point.

Design reference: "Detailed Architecture Diagram (Inference)" slide —
"Phase 2: AlphaRewrite Inference"

Algorithm
---------
  1. G = EGraphServer.egraph(T₀)
  2. while budget--:
  3.     p̂, v̂ = InferenceServer.predict(G)
  4.     pi, v = MCTSServer.PUCT(G, p̂, v̂)
  5.     action = argmax(pi)              # greedy at inference time
  6.     G = EGraphServer.apply(G, action)
  7. return EGraphServer.extract(G)      # → Optimized Term T_opt

Key differences from training self-play
----------------------------------------
* No replay buffer writes.
* Greedy action selection (temperature → 0 or argmax).
* Fewer MCTS simulations (latency-sensitive).
* Uses the same EGraphServer / InferenceServer / MCTSServer objects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OptimizedTerm:
    """Result of an optimization run.

    Attributes
    ----------
    original_term : str
        The input term / source code.
    optimized_term : str
        The optimised term extracted from the final e-graph.
    initial_cost : float
        Cost of the original term (instruction count or similar).
    final_cost : float
        Cost of the optimised term.
    cost_reduction_pct : float
        Percentage reduction: ``(initial - final) / initial * 100``.
    steps_taken : int
        Number of rewrite steps applied.
    elapsed_seconds : float
        Wall-clock time for the optimization.
    """
    original_term: str
    optimized_term: str
    initial_cost: float
    final_cost: float
    cost_reduction_pct: float
    steps_taken: int
    elapsed_seconds: float

    def __str__(self) -> str:
        return (
            f"OptimizedTerm(\n"
            f"  original  = {self.original_term!r}\n"
            f"  optimized = {self.optimized_term!r}\n"
            f"  cost      : {self.initial_cost:.2f} → {self.final_cost:.2f} "
            f"  ({self.cost_reduction_pct:+.1f}%)\n"
            f"  steps     = {self.steps_taken}\n"
            f"  elapsed   = {self.elapsed_seconds:.3f}s\n"
            f")"
        )


class AlphaRewriteOptimizer:
    """High-level optimizer that runs the full AlphaRewrite inference loop.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to a saved ``AlphaRewriteNet`` checkpoint.
    num_simulations : int
        MCTS simulations per step (lower = faster, less accurate).
    c_puct : float
        PUCT exploration constant.
    device : str
        PyTorch device.
    num_rules : int
        Size of the active rule set (must match checkpoint).
    embed_dim : int
        Network embedding dimension (must match checkpoint).

    Usage
    -----
    >>> opt = AlphaRewriteOptimizer("checkpoints/theta_k0019.pt")
    >>> result = opt.optimize("x * 2 + 0", budget=50)
    >>> print(result)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        num_simulations: int = 20,
        c_puct: float = 1.5,
        device: str = "cpu",
        num_rules: int = 64,
        embed_dim: int = 128,
    ) -> None:
        self.device = device

        # ---- Load network -----------------------------------------------
        from model.alpha_rewrite_net import AlphaRewriteNet
        from training.checkpoint import load_checkpoint

        self._net = AlphaRewriteNet(num_rules=num_rules, embed_dim=embed_dim)
        load_checkpoint(checkpoint_path, self._net, device=device)
        self._net.eval()

        # ---- Construct servers ------------------------------------------
        from data_generation.egraph_server import EGraphServer
        from data_generation.mcts import MCTSServer
        from data_generation.self_play import InferenceServer

        self._egraph_server = EGraphServer()
        self._mcts_server   = MCTSServer(
            egraph_server=self._egraph_server,
            c_puct=c_puct,
            num_simulations=num_simulations,
        )
        self._inference_server = InferenceServer(
            net=self._net,
            egraph_server=self._egraph_server,
            device=device,
        )

        logger.info(
            "AlphaRewriteOptimizer ready: checkpoint=%s, simulations=%d",
            checkpoint_path, num_simulations,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, term: str, budget: int = 50) -> OptimizedTerm:
        """Optimise *term* within *budget* rewrite steps.

        Parameters
        ----------
        term : str
            Source term / pseudo-code expression.
        budget : int
            Maximum number of rewrite steps.

        Returns
        -------
        OptimizedTerm
        """
        t0 = time.perf_counter()
        G  = self._egraph_server.egraph(term)
        initial_cost = self._egraph_server.cost(G)

        steps = 0
        while steps < budget and not self._egraph_server.is_terminal(G):
            prior, value = self._inference_server.predict(G)
            if not prior:
                break

            pi, _ = self._mcts_server.puct(
                egraph=G,
                prior_policy=prior,
                value_estimate=value,
                inference_fn=self._inference_server.predict,
            )

            # Greedy action selection at inference time.
            if not pi:
                break
            action = max(pi, key=lambda a: pi[a])
            G = self._egraph_server.apply(G, action)
            steps += 1

        optimized_term = self._egraph_server.extract(G)
        final_cost     = self._egraph_server.cost(G)
        elapsed        = time.perf_counter() - t0

        cost_reduction_pct = (
            (initial_cost - final_cost) / initial_cost * 100
            if initial_cost > 0 else 0.0
        )

        result = OptimizedTerm(
            original_term=term,
            optimized_term=optimized_term,
            initial_cost=initial_cost,
            final_cost=final_cost,
            cost_reduction_pct=cost_reduction_pct,
            steps_taken=steps,
            elapsed_seconds=elapsed,
        )
        logger.info("Optimization complete: %s", result)
        return result

    @classmethod
    def from_config(
        cls,
        checkpoint_path: str | Path,
        config: Any,          # TrainingConfig
    ) -> "AlphaRewriteOptimizer":
        """Convenience constructor using a :class:`~training.config.TrainingConfig`.

        Parameters
        ----------
        checkpoint_path : str or Path
        config : TrainingConfig
        """
        return cls(
            checkpoint_path=checkpoint_path,
            num_rules=config.num_rules,
            embed_dim=config.embed_dim,
            c_puct=config.c_puct,
            device=config.device,
        )

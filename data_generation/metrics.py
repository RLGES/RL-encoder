"""
data_generation/metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reward and statistics logging for self-play games.

Tracks per-game and aggregate statistics for monitoring training progress.
All metrics are logged via Python's standard ``logging`` module so they can
be captured by any configured handler (console, file, TensorBoard, W&B, etc.).

Example
-------
>>> metrics = GameMetrics()
>>> metrics.record_game(n_steps=15, initial_cost=42.0, final_cost=31.0, reward=0.26)
>>> metrics.log_summary()
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GameStats:
    """Per-game statistics record."""
    game_id: int
    n_steps: int
    initial_cost: float = 0.0
    final_cost: float = 0.0
    reward: float = 0.0       # z ∈ [-1, +1]
    n_samples: int = 0        # steps that produced training samples


class GameMetrics:
    """Accumulates and reports self-play performance metrics.

    Parameters
    ----------
    window : int
        Moving-average window size for recent-game stats.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.history: List[GameStats] = []

    # ------------------------------------------------------------------

    def record_game(
        self,
        n_steps: int,
        initial_cost: float = 0.0,
        final_cost: float = 0.0,
        reward: float = 0.0,
    ) -> None:
        """Record statistics for one completed game.

        Parameters
        ----------
        n_steps : int
            Number of rewrite steps taken.
        initial_cost : float
        final_cost : float
        reward : float
            Terminal reward z (normalised cost improvement).
        """
        game_id = len(self.history)
        stats = GameStats(
            game_id=game_id,
            n_steps=n_steps,
            initial_cost=initial_cost,
            final_cost=final_cost,
            reward=reward,
            n_samples=n_steps,
        )
        self.history.append(stats)
        logger.debug(
            "Game %d: steps=%d, cost %.2f → %.2f, reward=%.4f",
            game_id, n_steps, initial_cost, final_cost, reward,
        )

    def log_summary(self) -> None:
        """Log aggregate statistics over all recorded games."""
        if not self.history:
            logger.info("No games recorded yet.")
            return

        recent = self.history[-self.window:]
        rewards = [g.reward for g in recent]
        steps   = [g.n_steps for g in recent]

        logger.info(
            "Self-play summary [last %d games]: "
            "mean_reward=%.4f ± %.4f, mean_steps=%.1f, total_games=%d",
            len(recent),
            statistics.mean(rewards),
            statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            statistics.mean(steps),
            len(self.history),
        )

    @property
    def mean_reward(self) -> float:
        """Mean reward over the last ``window`` games."""
        recent = [g.reward for g in self.history[-self.window:]]
        return statistics.mean(recent) if recent else 0.0

    @property
    def total_samples(self) -> int:
        """Total number of training samples produced."""
        return sum(g.n_samples for g in self.history)

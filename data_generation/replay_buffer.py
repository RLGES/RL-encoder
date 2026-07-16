"""
data_generation/replay_buffer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`ReplayBuffer` — stores ``(EGraph, pi, z)`` samples from self-play
games and provides serialization to the canonical JSON schema for
debugging, logging, and inter-process transfer.

JSON record schema (one record per step in a game)
---------------------------------------------------
{
  "trajectory_id": str,
  "step_in_game": int,
  "state_G": {
    "description": str,          # human-readable term being rewritten
    "num_eclasses": int,
    "num_enodes": int
  },
  "mcts_policy_pi": { "<action_name>": float, ... },   # sums to ~1.0
  "target_value_z": float,       # filled in only after game terminates
  "cost_metadata": {
    "initial_cost": float,
    "final_cost": float,
    "cost_metric": str           # e.g. "instruction_count"
  }
}
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# On-disk record schema (matches spec exactly)
# ---------------------------------------------------------------------------

@dataclass
class StateSnapshot:
    description: str
    num_eclasses: int
    num_enodes: int


@dataclass
class CostMetadata:
    initial_cost: float
    final_cost: float
    cost_metric: str = "instruction_count"


@dataclass
class TrajectoryRecord:
    """One row in the replay buffer, JSON-serialisable.

    Matches the schema specified in the design document.
    """
    trajectory_id: str
    step_in_game: int
    state_G: StateSnapshot
    mcts_policy_pi: Dict[str, float]        # action_name -> probability
    target_value_z: float                   # retroactively filled after game ends
    cost_metadata: CostMetadata

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryRecord":
        return cls(
            trajectory_id=d["trajectory_id"],
            step_in_game=d["step_in_game"],
            state_G=StateSnapshot(**d["state_G"]),
            mcts_policy_pi=d["mcts_policy_pi"],
            target_value_z=d["target_value_z"],
            cost_metadata=CostMetadata(**d["cost_metadata"]),
        )


# ---------------------------------------------------------------------------
# In-memory buffer
# ---------------------------------------------------------------------------

# Type alias: each raw sample is (EGraph, policy_dict, z)
RawSample = Tuple[Any, Dict[Any, float], float]


class ReplayBuffer:
    """Fixed-capacity circular buffer of ``(EGraph, pi, z)`` training samples.

    Parameters
    ----------
    capacity : int
        Maximum number of samples to retain.  Oldest samples are evicted
        once capacity is reached.
    cost_metric : str
        Label for the cost metric used in serialized records.

    Usage
    -----
    >>> buf = ReplayBuffer(capacity=10_000)
    >>> buf.extend(game_samples)
    >>> batch = buf.sample(batch_size=64)
    >>> buf.save("replay_k0.jsonl")
    >>> buf.load("replay_k0.jsonl")
    """

    def __init__(
        self,
        capacity: int = 100_000,
        cost_metric: str = "instruction_count",
    ) -> None:
        self.capacity = capacity
        self.cost_metric = cost_metric
        self._buffer: List[RawSample] = []
        self._records: List[TrajectoryRecord] = []   # parallel list for serialization
        self._ptr: int = 0                            # write pointer (circular)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def push(
        self,
        egraph: Any,
        pi: Dict[Any, float],
        z: float,
        trajectory_id: Optional[str] = None,
        step_in_game: int = 0,
        initial_cost: float = 0.0,
        final_cost: float = 0.0,
    ) -> None:
        """Add one ``(EGraph, pi, z)`` sample.

        Parameters
        ----------
        egraph : EGraph
        pi : Dict[Action, float]
        z : float
        trajectory_id : str, optional
            UUID for the game this step belongs to.
        step_in_game : int
        initial_cost, final_cost : float
        """
        sample: RawSample = (egraph, pi, z)
        record = self._make_record(
            egraph=egraph, pi=pi, z=z,
            trajectory_id=trajectory_id or str(uuid.uuid4()),
            step_in_game=step_in_game,
            initial_cost=initial_cost,
            final_cost=final_cost,
        )

        if len(self._buffer) < self.capacity:
            self._buffer.append(sample)
            self._records.append(record)
        else:
            self._buffer[self._ptr] = sample
            self._records[self._ptr] = record
        self._ptr = (self._ptr + 1) % self.capacity

    def extend(
        self,
        samples: List[RawSample],
        trajectory_id: Optional[str] = None,
    ) -> None:
        """Push a list of samples (e.g. output of ``play_game``)."""
        tid = trajectory_id or str(uuid.uuid4())
        for step, (egraph, pi, z) in enumerate(samples):
            self.push(
                egraph=egraph, pi=pi, z=z,
                trajectory_id=tid, step_in_game=step,
            )

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> List[RawSample]:
        """Uniformly sample *batch_size* entries (with replacement).

        Returns
        -------
        List[RawSample]
        """
        import random
        if len(self._buffer) == 0:
            return []
        return random.choices(self._buffer, k=batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[RawSample]:
        return iter(self._buffer)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the metadata records to a JSON Lines file.

        Note: EGraph objects are *not* serialised (they may be large).
        Only the ``TrajectoryRecord`` schema is written.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict()) + "\n")
        logger.info("Saved %d records to %s", len(self._records), path)

    def load(self, path: str | Path) -> None:
        """Load records from a JSON Lines file (metadata only; no EGraphs)."""
        path = Path(path)
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = TrajectoryRecord.from_dict(json.loads(line))
                    self._records.append(record)
                    # Push a placeholder raw sample (EGraph = None for loaded records).
                    self._buffer.append((None, record.mcts_policy_pi, record.target_value_z))
        logger.info("Loaded %d records from %s", len(self._records), path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_record(
        self,
        egraph: Any,
        pi: Dict[Any, float],
        z: float,
        trajectory_id: str,
        step_in_game: int,
        initial_cost: float,
        final_cost: float,
    ) -> TrajectoryRecord:
        """Build a JSON-serialisable record from raw sample data."""
        # Extract EGraph metadata (number of e-classes / enodes).
        try:
            num_eclasses = len(egraph.eclasses)
            num_enodes   = sum(len(ec.nodes) for ec in egraph.eclasses.values())
            description  = getattr(egraph, "description", str(egraph))
        except Exception:
            num_eclasses = 0
            num_enodes   = 0
            description  = "<unknown>"

        # Serialise policy keys as strings.
        pi_str: Dict[str, float] = {str(a): float(p) for a, p in pi.items()}

        return TrajectoryRecord(
            trajectory_id=trajectory_id,
            step_in_game=step_in_game,
            state_G=StateSnapshot(
                description=description,
                num_eclasses=num_eclasses,
                num_enodes=num_enodes,
            ),
            mcts_policy_pi=pi_str,
            target_value_z=float(z),
            cost_metadata=CostMetadata(
                initial_cost=initial_cost,
                final_cost=final_cost,
                cost_metric=self.cost_metric,
            ),
        )

"""
tests/test_data_generation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Smoke tests for Module 3: data_generation components.

Tests
-----
* test_replay_buffer_push_pop  : push N samples, verify length and schema
* test_replay_buffer_serialise : save → load round-trip preserves records
* test_replay_buffer_sample    : sample returns correct batch size
* test_mcts_node               : MCTSNode initialises with correct defaults
* test_game_metrics            : GameMetrics records and reports correctly
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from data_generation.metrics import GameMetrics
from data_generation.mcts import MCTSNode
from data_generation.replay_buffer import ReplayBuffer, TrajectoryRecord


class TestReplayBuffer:
    def test_push_and_length(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(egraph=None, pi={"action_a": 1.0}, z=0.5,
                     trajectory_id="traj-0", step_in_game=i)
        assert len(buf) == 10

    def test_capacity_eviction(self) -> None:
        buf = ReplayBuffer(capacity=5)
        for i in range(8):
            buf.push(egraph=None, pi={"a": 1.0}, z=float(i),
                     trajectory_id="t", step_in_game=i)
        assert len(buf) == 5  # evicted 3 oldest

    def test_sample_size(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(egraph=None, pi={"a": 1.0}, z=0.0,
                     trajectory_id="t", step_in_game=i)
        batch = buf.sample(7)
        assert len(batch) == 7

    def test_sample_empty(self) -> None:
        buf = ReplayBuffer(capacity=100)
        assert buf.sample(10) == []

    def test_serialise_round_trip(self) -> None:
        buf = ReplayBuffer(capacity=100)
        buf.push(
            egraph=None,
            pi={"rule_add_zero": 0.6, "rule_mul_one": 0.4},
            z=0.25,
            trajectory_id="abc-123",
            step_in_game=0,
            initial_cost=10.0,
            final_cost=7.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test_replay.jsonl"
            buf.save(p)

            # Verify JSON Lines format is valid.
            with p.open() as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["trajectory_id"] == "abc-123"
            assert record["step_in_game"] == 0
            assert "state_G" in record
            assert "mcts_policy_pi" in record
            assert "target_value_z" in record
            assert "cost_metadata" in record

            # Load back.
            buf2 = ReplayBuffer(capacity=100)
            buf2.load(p)
            assert len(buf2) == 1

    def test_schema_fields(self) -> None:
        """TrajectoryRecord must have all required schema keys."""
        buf = ReplayBuffer()
        buf.push(egraph=None, pi={"a": 1.0}, z=0.0, trajectory_id="x", step_in_game=0)
        record = buf._records[0]
        d = record.to_dict()
        required_keys = {
            "trajectory_id", "step_in_game", "state_G",
            "mcts_policy_pi", "target_value_z", "cost_metadata"
        }
        assert required_keys.issubset(d.keys()), \
            f"Missing keys: {required_keys - d.keys()}"


class TestMCTSNode:
    def test_default_initialisation(self) -> None:
        node = MCTSNode(egraph="dummy", prior=0.5)
        assert node.visit_count == 0
        assert node.is_leaf()
        assert not node.is_expanded()
        assert node.prior == pytest.approx(0.5)
        assert node.children == {}


class TestGameMetrics:
    def test_record_and_mean(self) -> None:
        m = GameMetrics(window=10)
        m.record_game(n_steps=5, initial_cost=10.0, final_cost=8.0, reward=0.2)
        m.record_game(n_steps=3, initial_cost=10.0, final_cost=9.0, reward=0.1)
        assert abs(m.mean_reward - 0.15) < 1e-6
        assert m.total_samples == 8

    def test_log_summary_no_crash(self) -> None:
        m = GameMetrics()
        m.record_game(n_steps=1)
        m.log_summary()  # must not raise

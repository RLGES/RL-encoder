"""
data_generation — Self-play data collection pipeline.

Components
----------
  EGraphServer    : wraps Learn-R's e-graph + rule engine
  InferenceServer : wraps AlphaRewriteNet for prediction
  MCTSServer      : PUCT-based Monte Carlo Tree Search
  ReplayBuffer    : stores and persists (G, pi, v) tuples
  Metrics         : reward & stats logging utilities
  play_game       : core self-play loop
"""

from data_generation.egraph_server import EGraphServer
from data_generation.mcts import MCTSServer, MCTSNode
from data_generation.replay_buffer import ReplayBuffer
from data_generation.self_play import play_game

__all__ = [
    "EGraphServer",
    "MCTSServer",
    "MCTSNode",
    "ReplayBuffer",
    "play_game",
]

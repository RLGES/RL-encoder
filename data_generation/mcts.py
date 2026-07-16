"""
data_generation/mcts.py
~~~~~~~~~~~~~~~~~~~~~~~~
Monte Carlo Tree Search with PUCT selection for the e-graph optimization game.

Design reference: "PLAY GAME OPTIMIZATION ALGORITHM" slide

Algorithm phases per MCTS call
-------------------------------
  1. **Selection**   : traverse from root using PUCT until a leaf is reached.
  2. **Expansion**   : expand the leaf by enumerating legal actions from EGraphServer.
  3. **Simulation**  : call InferenceServer.predict(G) to get (p̂, v̂).
  4. **Backprop**    : propagate v̂ up the path to root, updating N and Q.

PUCT formula (UCB for trees with a prior)
------------------------------------------
  PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

where:
  * Q(s, a) : mean action-value (running average of backed-up v̂)
  * P(s, a) : prior probability from InferenceServer policy head
  * N(s)    : total visit count of node s
  * N(s, a) : visit count of edge (s, a)
  * c_puct  : exploration constant (TrainingConfig.c_puct)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from data_generation.egraph_server import Action, EGraphServer


# ---------------------------------------------------------------------------
# MCTS Tree Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """A single node in the MCTS search tree.

    Attributes
    ----------
    egraph : EGraph
        The e-graph state represented by this node.
    parent : MCTSNode, optional
        Parent node (None for root).
    action_from_parent : Action, optional
        The action that produced this node from its parent.
    prior : float
        Prior probability P(s_parent, a) from the policy head.
    children : Dict[Action, MCTSNode]
        Expanded child nodes keyed by action.
    visit_count : int
        N(s) — total visits to this node.
    action_counts : Dict[Action, int]
        N(s, a) — visit count per edge.
    action_values : Dict[Action, float]
        Q(s, a) — mean backed-up value per edge.
    """

    def __init__(
        self,
        egraph: Any,
        parent: Optional["MCTSNode"] = None,
        action_from_parent: Optional[Action] = None,
        prior: float = 1.0,
    ) -> None:
        self.egraph = egraph
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = prior

        self.children: Dict[Action, MCTSNode] = {}
        self.visit_count: int = 0
        self.action_counts: Dict[Action, int] = {}
        self.action_values: Dict[Action, float] = {}

    def is_expanded(self) -> bool:
        """True once children have been populated."""
        return len(self.children) > 0

    def is_leaf(self) -> bool:
        """True for unexpanded nodes."""
        return not self.is_expanded()

    def __repr__(self) -> str:
        return (
            f"MCTSNode(visits={self.visit_count}, "
            f"children={len(self.children)}, "
            f"prior={self.prior:.3f})"
        )


# ---------------------------------------------------------------------------
# MCTS Server
# ---------------------------------------------------------------------------

class MCTSServer:
    """PUCT-based Monte Carlo Tree Search for e-graph rewriting.

    Parameters
    ----------
    egraph_server : EGraphServer
        Provides ``legal_actions`` and ``apply`` methods.
    c_puct : float
        Exploration constant in the PUCT formula.
    num_simulations : int
        Number of MCTS simulations to run per ``puct()`` call.

    Usage
    -----
    >>> mcts = MCTSServer(egraph_server, c_puct=1.5, num_simulations=50)
    >>> pi, v = mcts.puct(egraph_state, prior_policy, value_estimate)
    """

    def __init__(
        self,
        egraph_server: EGraphServer,
        c_puct: float = 1.5,
        num_simulations: int = 50,
    ) -> None:
        self.egraph_server = egraph_server
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def puct(
        self,
        egraph: Any,
        prior_policy: Dict[Action, float],
        value_estimate: float,
        inference_fn: Any,
    ) -> Tuple[Dict[Action, float], float]:
        """Run MCTS from *egraph* and return improved policy *pi* and *v*.

        Parameters
        ----------
        egraph : EGraph
            Current game state.
        prior_policy : Dict[Action, float]
            Initial policy from :class:`~data_generation.self_play.InferenceServer`.
            Maps Action -> probability.
        value_estimate : float
            Initial value estimate ``v̂`` from the inference server.
        inference_fn : callable
            Callable ``(EGraph) -> (Dict[Action, float], float)`` used at
            expansion time to get prior + value for newly visited states.

        Returns
        -------
        pi : Dict[Action, float]
            Improved policy (visit-count proportional distribution).
        v  : float
            Backed-up value at root after simulations.
        """
        root = MCTSNode(egraph=egraph, prior=1.0)
        # Seed root with prior policy.
        self._expand_node(root, prior_policy)

        for _ in range(self.num_simulations):
            path, leaf = self._select(root)
            if self.egraph_server.is_terminal(leaf.egraph):
                leaf_value = self._terminal_value(leaf.egraph)
            else:
                prior, leaf_value = inference_fn(leaf.egraph)
                if leaf.is_leaf():
                    self._expand_node(leaf, prior)
            self._backprop(path, leaf_value)

        # Build improved policy from visit counts.
        pi = self._visit_count_policy(root)
        v = self._root_value(root)
        return pi, v

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> Tuple[List[Tuple[MCTSNode, Action]], MCTSNode]:
        """Selection: descend tree using PUCT until a leaf is reached.

        Returns
        -------
        path : List[(node, action)]
            The sequence of (parent, chosen_action) pairs from root to leaf.
        leaf : MCTSNode
            The unexpanded leaf node.
        """
        # TODO: implement full PUCT traversal.
        path: List[Tuple[MCTSNode, Action]] = []
        current = node
        while current.is_expanded():
            action = self._puct_select_action(current)
            if action not in current.children:
                break
            path.append((current, action))
            current = current.children[action]
        return path, current

    def _expand_node(self, node: MCTSNode, prior_policy: Dict[Action, float]) -> None:
        """Expansion: create child nodes for all legal actions.

        Parameters
        ----------
        node : MCTSNode
        prior_policy : Dict[Action, float]
            Prior probabilities from the network's policy head.
        """
        # TODO: handle edge case where prior_policy and legal_actions disagree.
        legal = self.egraph_server.legal_actions(node.egraph)
        for action in legal:
            prior = prior_policy.get(action, 1.0 / max(len(legal), 1))
            child_egraph = self.egraph_server.apply(node.egraph, action)
            child = MCTSNode(
                egraph=child_egraph,
                parent=node,
                action_from_parent=action,
                prior=prior,
            )
            node.children[action] = child
            node.action_counts[action] = 0
            node.action_values[action] = 0.0

    def _backprop(
        self, path: List[Tuple[MCTSNode, Action]], value: float
    ) -> None:
        """Backpropagation: update Q and N along *path*.

        Parameters
        ----------
        path : List[(MCTSNode, Action)]
        value : float
            Value to propagate (from leaf simulation or terminal outcome).
        """
        # TODO: implement sign flip (value from opponent's perspective).
        for node, action in reversed(path):
            node.visit_count += 1
            node.action_counts[action] = node.action_counts.get(action, 0) + 1
            old_q = node.action_values.get(action, 0.0)
            n = node.action_counts[action]
            node.action_values[action] = old_q + (value - old_q) / n

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _puct_select_action(self, node: MCTSNode) -> Action:
        """Choose the action maximising the PUCT score.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        # TODO: add Dirichlet noise at root for exploration during training.
        best_score = float("-inf")
        best_action: Optional[Action] = None
        sqrt_total = math.sqrt(max(node.visit_count, 1))
        for action, child in node.children.items():
            q = node.action_values.get(action, 0.0)
            n_a = node.action_counts.get(action, 0)
            p = child.prior
            score = q + self.c_puct * p * sqrt_total / (1 + n_a)
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            raise RuntimeError("PUCT selection failed: no children in expanded node")
        return best_action

    def _visit_count_policy(self, root: MCTSNode) -> Dict[Action, float]:
        """Convert visit counts to a probability distribution.

        Returns
        -------
        pi : Dict[Action, float]  — sums to 1.0
        """
        total = sum(root.action_counts.values()) or 1
        return {a: n / total for a, n in root.action_counts.items()}

    def _terminal_value(self, egraph: Any) -> float:
        """Return value for a terminal state (fully saturated e-graph)."""
        # TODO: compute normalised cost improvement from final extracted term.
        return 0.0

    def _root_value(self, root: MCTSNode) -> float:
        """Aggregate backed-up value at the root (mean Q across actions)."""
        if not root.action_values:
            return 0.0
        return sum(root.action_values.values()) / len(root.action_values)

"""
data_generation/egraph_server.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`EGraphServer` — adapter that wraps Learn-R's e-graph engine and
exposes the three verbs used by the self-play loop:

  * ``egraph(T)``        : build an EGraph from a source term ``T``
  * ``apply(G, action)`` : apply a (rule, eclass) action, return updated EGraph
  * ``extract(G)``       : extract the lowest-cost term from ``G``

This is **not** a real network server; it is simply a Python object that
encapsulates the Learn-R pipeline behind a stable interface so the RL
training code is independent of pipeline internals.

Design reference: "PHASE 2: RL OPTIMIZATION ENGINE" slide
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import Learn-R internals (guarded for isolated testing)
# ---------------------------------------------------------------------------
try:
    from egraph_bridge.simple_egraph import EGraph  # type: ignore[import]
except ImportError:  # pragma: no cover
    EGraph = Any  # type: ignore[assignment,misc]

try:
    from hierarchical_engine.engine import RewriteEngine  # type: ignore[import]
except ImportError:  # pragma: no cover
    RewriteEngine = Any  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Action type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """A single rewriting action: apply ``rule_id`` to ``eclass_id``.

    Attributes
    ----------
    rule_id : int
        Index into the Active Rule Set (matching RuleEmbeddingTable).
    rule_name : str
        Human-readable rule name (for logging).
    eclass_id : int
        Canonical e-class ID in the current EGraph to rewrite.
    """
    rule_id: int
    rule_name: str
    eclass_id: int

    def __str__(self) -> str:
        return f"Action(rule={self.rule_name!r}, eclass={self.eclass_id})"


# ---------------------------------------------------------------------------
# EGraphServer
# ---------------------------------------------------------------------------

class EGraphServer:
    """Facade over Learn-R's e-graph construction and rule-application engine.

    Parameters
    ----------
    rule_engine : RewriteEngine, optional
        A pre-initialised Learn-R rewrite engine.  If ``None``, a stub is used
        (useful for smoke testing without the full Learn-R stack).
    cost_fn : callable, optional
        Function ``(EGraph) -> float`` that measures program cost.
        Defaults to instruction-count heuristic.

    Example
    -------
    >>> server = EGraphServer()
    >>> G = server.egraph("x * 2")
    >>> actions = server.legal_actions(G)
    >>> G2 = server.apply(G, actions[0])
    >>> term = server.extract(G2)
    """

    def __init__(
        self,
        rule_engine: Optional["RewriteEngine"] = None,
        cost_fn: Optional[Any] = None,
    ) -> None:
        self.rule_engine = rule_engine
        self.cost_fn = cost_fn or self._default_cost

    # ------------------------------------------------------------------
    # Public interface (used by self_play.py)
    # ------------------------------------------------------------------

    def egraph(self, term: str) -> "EGraph":
        """Parse *term* and construct an initial EGraph.

        Parameters
        ----------
        term : str
            Source term / pseudo-code to optimise (e.g. ``"x * 2 + 0"``).

        Returns
        -------
        EGraph
            Freshly built e-graph with a single e-class per sub-expression.
        """
        # TODO: integrate with Learn-R's frontend / SSA pipeline.
        #       Stub: return a minimal dummy EGraph for smoke testing.
        try:
            from egraph_bridge.simple_egraph import EGraph as _EGraph  # type: ignore[import]
            from egraph_bridge.expr_to_egraph import expr_to_egraph    # type: ignore[import]
            return expr_to_egraph(term)
        except (ImportError, Exception):  # pragma: no cover
            return _DummyEGraph(description=term)  # type: ignore[return-value]

    def apply(self, egraph: "EGraph", action: Action) -> "EGraph":
        """Apply *action* to *egraph* and return the updated EGraph.

        The original EGraph is **not** mutated; a new object (or shallow copy
        with the relevant e-class merged) is returned.

        Parameters
        ----------
        egraph : EGraph
        action : Action

        Returns
        -------
        EGraph
            Updated e-graph after the rewrite.
        """
        # TODO: delegate to self.rule_engine.apply_rule(egraph, action.rule_name, action.eclass_id)
        #       Stub: return the same egraph unchanged.
        return egraph

    def extract(self, egraph: "EGraph") -> str:
        """Extract the minimum-cost term from *egraph*.

        Uses the extraction algorithm from Learn-R (cost-minimising tree
        extraction over the e-graph).

        Returns
        -------
        str
            String representation of the optimised term.
        float
            Final cost of the extracted term.
        """
        # TODO: call egraph_bridge extraction with self.cost_fn.
        return getattr(egraph, "description", "<extracted_term>")

    def legal_actions(self, egraph: "EGraph") -> List[Action]:
        """Enumerate all currently applicable actions on *egraph*.

        Returns
        -------
        List[Action]
            All (rule, eclass) pairs where the rule LHS matches some enode in
            the corresponding e-class.
        """
        # TODO: iterate active rule set, run matcher against egraph.
        return []

    def cost(self, egraph: "EGraph") -> float:
        """Compute the current best extractable cost from *egraph*."""
        return self.cost_fn(egraph)

    def is_terminal(self, egraph: "EGraph") -> bool:
        """Return True when no further legal actions exist (saturated graph)."""
        return len(self.legal_actions(egraph)) == 0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _default_cost(egraph: "EGraph") -> float:
        """Placeholder: count total enodes as a proxy for instruction cost."""
        # TODO: replace with proper extraction-based cost.
        try:
            total = sum(len(ec.nodes) for ec in egraph.eclasses.values())
            return float(total)
        except AttributeError:
            return 1.0


# ---------------------------------------------------------------------------
# Dummy EGraph for smoke tests (no Learn-R dependency)
# ---------------------------------------------------------------------------

class _DummyEGraph:
    """Minimal stand-in for EGraph when Learn-R is not installed."""

    def __init__(self, description: str = "dummy") -> None:
        self.description = description
        self.eclasses: Dict[int, Any] = {}

    def _find(self, eid: int) -> int:  # noqa: D401
        return eid

"""
inference — AlphaRewrite Inference (Phase 2).

Components
----------
  AlphaRewriteOptimizer : high-level optimizer object
  CLI                   : ``python -m inference.cli``
"""

from inference.optimizer import AlphaRewriteOptimizer, OptimizedTerm

__all__ = ["AlphaRewriteOptimizer", "OptimizedTerm"]

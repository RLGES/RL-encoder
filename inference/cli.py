"""
inference/cli.py
~~~~~~~~~~~~~~~~
Command-line interface for AlphaRewrite inference.

Usage
-----
  python -m inference.cli --checkpoint checkpoints/theta_k0019.pt \\
                          --input test_code.txt --budget 50

Or directly:
  python -m inference.cli --term "x * 2 + 0" --checkpoint path/to/ckpt.pt

If both ``--input`` and ``--term`` are given, ``--term`` takes precedence.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inference.cli",
        description="AlphaRewrite: RL-guided equality-saturation optimizer",
    )
    p.add_argument(
        "--checkpoint", "-c", required=True, type=Path,
        help="Path to AlphaRewriteNet checkpoint file (e.g. checkpoints/theta_k0019.pt)",
    )
    p.add_argument(
        "--input", "-i", type=Path, default=None,
        help="Path to a text file containing the term/expression to optimize.",
    )
    p.add_argument(
        "--term", "-t", type=str, default=None,
        help="Inline term to optimize (overrides --input).",
    )
    p.add_argument(
        "--budget", "-b", type=int, default=50,
        help="Maximum number of rewrite steps (default: 50).",
    )
    p.add_argument(
        "--simulations", "-s", type=int, default=20,
        help="MCTS simulations per step (default: 20).",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: cpu).",
    )
    p.add_argument(
        "--num-rules", type=int, default=64,
        help="Active rule set size (must match checkpoint; default: 64).",
    )
    p.add_argument(
        "--embed-dim", type=int, default=128,
        help="Network embedding dimension (must match checkpoint; default: 128).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ---- Resolve term ---------------------------------------------------
    if args.term is not None:
        term = args.term
    elif args.input is not None:
        if not args.input.exists():
            logger.error("Input file not found: %s", args.input)
            return 1
        term = args.input.read_text().strip()
    else:
        logger.error("Provide --term or --input.")
        parser.print_help()
        return 1

    if not term:
        logger.error("Input term is empty.")
        return 1

    logger.info("Optimizing: %r", term)
    logger.info("Budget: %d, Simulations: %d, Device: %s", args.budget, args.simulations, args.device)

    # ---- Load optimizer -------------------------------------------------
    try:
        from inference.optimizer import AlphaRewriteOptimizer
        optimizer = AlphaRewriteOptimizer(
            checkpoint_path=args.checkpoint,
            num_simulations=args.simulations,
            device=args.device,
            num_rules=args.num_rules,
            embed_dim=args.embed_dim,
        )
    except Exception as exc:
        logger.error("Failed to load optimizer: %s", exc)
        return 1

    # ---- Run inference --------------------------------------------------
    try:
        result = optimizer.optimize(term=term, budget=args.budget)
    except Exception as exc:
        logger.error("Optimization failed: %s", exc)
        return 1

    # ---- Print result ---------------------------------------------------
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

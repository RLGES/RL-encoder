"""
tests/test_inference.py
~~~~~~~~~~~~~~~~~~~~~~~~
Smoke tests for Module 5: inference optimizer and CLI.

Tests
-----
* test_optimized_term_dataclass   : OptimizedTerm fields and str() formatting
* test_cli_parser_no_crash        : argument parser builds without error
* test_cli_missing_checkpoint     : CLI exits with error code 1 for missing ckpt
"""

from __future__ import annotations

import pytest

from inference.optimizer import OptimizedTerm


class TestOptimizedTerm:
    def test_fields(self) -> None:
        result = OptimizedTerm(
            original_term="x * 2 + 0",
            optimized_term="x * 2",
            initial_cost=10.0,
            final_cost=8.0,
            cost_reduction_pct=20.0,
            steps_taken=3,
            elapsed_seconds=0.042,
        )
        assert result.original_term   == "x * 2 + 0"
        assert result.optimized_term  == "x * 2"
        assert result.initial_cost    == pytest.approx(10.0)
        assert result.final_cost      == pytest.approx(8.0)
        assert result.cost_reduction_pct == pytest.approx(20.0)
        assert result.steps_taken     == 3

    def test_str_representation(self) -> None:
        result = OptimizedTerm(
            original_term="a", optimized_term="b",
            initial_cost=5.0, final_cost=4.0,
            cost_reduction_pct=20.0, steps_taken=1,
            elapsed_seconds=0.001,
        )
        s = str(result)
        assert "original" in s
        assert "optimized" in s
        assert "20.0" in s


class TestCLIParser:
    def test_parser_builds(self) -> None:
        from inference.cli import build_parser
        parser = build_parser()
        assert parser is not None

    def test_cli_missing_checkpoint(self) -> None:
        """CLI must exit with code 1 when the checkpoint file does not exist."""
        from inference.cli import main
        exit_code = main([
            "--checkpoint", "/nonexistent/path/ckpt.pt",
            "--term", "x + 0",
            "--budget", "2",
        ])
        assert exit_code == 1

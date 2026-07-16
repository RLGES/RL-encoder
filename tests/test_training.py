"""
tests/test_training.py
~~~~~~~~~~~~~~~~~~~~~~~
Smoke tests for Module 4: training utilities.

Tests
-----
* test_alpha_zero_loss_shapes      : loss returns 3 scalar tensors
* test_alpha_zero_loss_soft_labels : loss decreases after one gradient step
* test_loss_with_logits            : correctly handles pre-softmax logits
* test_config_defaults             : TrainingConfig instantiates without error
"""

from __future__ import annotations

import pytest
import torch

from training.config import TrainingConfig
from training.loss import alpha_zero_loss, policy_cross_entropy


class TestAlphaZeroLoss:
    def _make_batch(self, B: int, A: int):
        pi_target = torch.rand(B, A)
        pi_target /= pi_target.sum(dim=-1, keepdim=True)
        pi_pred   = torch.rand(B, A)
        v_pred    = torch.rand(B) * 2 - 1
        z         = torch.rand(B) * 2 - 1
        return pi_pred, pi_target, v_pred, z

    def test_returns_three_scalars(self) -> None:
        pi_pred, pi_target, v, z = self._make_batch(B=4, A=8)
        total, policy, value = alpha_zero_loss(pi_pred, pi_target, v, z)
        for name, t in [("total", total), ("policy", policy), ("value", value)]:
            assert t.shape == (), f"{name} is not scalar: {t.shape}"
            assert torch.isfinite(t), f"{name} is not finite: {t}"

    def test_loss_is_non_negative(self) -> None:
        pi_pred, pi_target, v, z = self._make_batch(B=8, A=4)
        total, _, _ = alpha_zero_loss(pi_pred, pi_target, v, z)
        assert total.item() >= 0.0, f"Total loss is negative: {total.item()}"

    def test_gradient_step_reduces_loss(self) -> None:
        """A single gradient step should reduce the loss."""
        A = 6
        pi_pred_param = torch.nn.Parameter(torch.randn(1, A))
        opt = torch.optim.SGD([pi_pred_param], lr=0.5)

        pi_target = torch.zeros(1, A)
        pi_target[0, 0] = 1.0  # one-hot
        v = torch.zeros(1)
        z = torch.zeros(1)

        total_before, _, _ = alpha_zero_loss(
            torch.softmax(pi_pred_param, dim=-1), pi_target, v, z
        )
        opt.zero_grad()
        loss, _, _ = alpha_zero_loss(
            torch.softmax(pi_pred_param, dim=-1), pi_target, v, z
        )
        loss.backward()
        opt.step()

        with torch.no_grad():
            total_after, _, _ = alpha_zero_loss(
                torch.softmax(pi_pred_param, dim=-1), pi_target, v, z
            )
        assert total_after.item() <= total_before.item() + 1e-4, \
            f"Loss did not decrease: {total_before.item():.4f} → {total_after.item():.4f}"

    def test_logits_input(self) -> None:
        """alpha_zero_loss must handle raw logits (values outside [0, 1])."""
        pi_pred   = torch.randn(3, 5) * 10.0   # clearly logits
        pi_target = torch.rand(3, 5)
        pi_target /= pi_target.sum(dim=-1, keepdim=True)
        v, z = torch.zeros(3), torch.zeros(3)
        total, _, _ = alpha_zero_loss(pi_pred, pi_target, v, z)
        assert torch.isfinite(total)

    def test_policy_cross_entropy_standalone(self) -> None:
        A = 4
        log_pi    = torch.log_softmax(torch.randn(A), dim=0)
        pi_target = torch.zeros(A); pi_target[0] = 1.0
        ce = policy_cross_entropy(log_pi, pi_target)
        assert ce.shape == ()
        assert ce.item() >= 0.0


class TestTrainingConfig:
    def test_default_instantiation(self) -> None:
        config = TrainingConfig()
        assert config.embed_dim == 128
        assert config.K_rounds == 20
        assert config.n_games  == 100
        assert len(config.training_terms) >= 1

    def test_override(self) -> None:
        config = TrainingConfig(embed_dim=64, lr=1e-3)
        assert config.embed_dim == 64
        assert config.lr == pytest.approx(1e-3)

"""
training/train_loop.py
~~~~~~~~~~~~~~~~~~~~~~~
AlphaZero-style outer training loop for EqSat RL.

Design reference: "Detailed Architecture Diagram (Training)" slide

Algorithm (outer loop)
-----------------------
  θ_0 = initialize_parameters()
  publish(θ_0)                       # → InferenceServer
  for k in 0..K-1:
      θ_gen = current weights
      D_k   = []
      for _ in range(N_games):
          samples = play_game(θ_gen)
          D_k.extend(samples)
      θ_{k+1} = train(D_k, T_steps)  # gradient descent on AlphaZero loss
      publish(θ_{k+1})               # → InferenceServer

Gradient step (inner loop)
---------------------------
  batch = D_k.sample(batch_size)
  pi_pred, v_pred = net(G_batch)
  loss = alpha_zero_loss(pi_pred, pi_target, v_pred, z)
  loss.backward(); optimizer.step()
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.optim as optim

from model.alpha_rewrite_net import AlphaRewriteNet
from training.checkpoint import publish_to_inference_server, save_checkpoint
from training.config import TrainingConfig
from training.loss import alpha_zero_loss

logger = logging.getLogger(__name__)


def initialize_parameters(config: TrainingConfig) -> AlphaRewriteNet:
    """Construct and randomly-initialise :class:`AlphaRewriteNet`.

    Parameters
    ----------
    config : TrainingConfig

    Returns
    -------
    AlphaRewriteNet
        Network with randomly initialised weights ``θ_0``.
    """
    net = AlphaRewriteNet(
        embed_dim=config.embed_dim,
        attn_dim=config.attn_dim,
        num_gnn_layers=config.num_gnn_layers,
        tau=config.tau,
        num_rules=config.num_rules,
        opcode_vocab_size=config.opcode_vocab_size,
    )
    net.to(config.device)
    logger.info(
        "Initialised AlphaRewriteNet: embed_dim=%d, layers=%d, rules=%d",
        config.embed_dim, config.num_gnn_layers, config.num_rules,
    )
    return net


def _train_one_round(
    net: AlphaRewriteNet,
    replay_buffer: "ReplayBuffer",    # type: ignore[name-defined]
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    round_idx: int,
) -> dict:
    """Run T_steps gradient updates on the replay buffer.

    Parameters
    ----------
    net : AlphaRewriteNet
    replay_buffer : ReplayBuffer
    optimizer : Optimizer
    config : TrainingConfig
    round_idx : int
        Current outer-loop round (for logging).

    Returns
    -------
    dict
        Summary statistics: mean_loss, mean_policy_loss, mean_value_loss.
    """
    net.train()
    total_loss   = 0.0
    total_policy = 0.0
    total_value  = 0.0
    steps_done   = 0

    for step in range(config.T_steps):
        batch = replay_buffer.sample(config.batch_size)
        if not batch:
            logger.warning("Replay buffer empty; skipping training step %d.", step)
            break

        # TODO: build real batched tensors from (EGraph, pi, z) tuples.
        #       Stub: use random tensors for smoke-test plumbing.
        pi_target = torch.rand(config.batch_size, config.num_rules)
        pi_target /= pi_target.sum(dim=-1, keepdim=True)
        pi_pred   = torch.rand(config.batch_size, config.num_rules)
        v_pred    = torch.rand(config.batch_size) * 2 - 1  # uniform in [-1, 1]
        z         = torch.rand(config.batch_size) * 2 - 1

        pi_pred   = pi_pred.to(config.device)
        pi_target = pi_target.to(config.device)
        v_pred    = v_pred.to(config.device)
        z         = z.to(config.device)

        loss, policy_loss, value_loss = alpha_zero_loss(
            pi_pred=pi_pred,
            pi_target=pi_target,
            v_pred=v_pred,
            z=z,
            l2_reg=config.weight_decay,
            model_params=list(net.parameters()),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
        optimizer.step()

        total_loss   += loss.item()
        total_policy += policy_loss.item()
        total_value  += value_loss.item()
        steps_done   += 1

        if step % config.log_every == 0:
            logger.info(
                "Round %d | Step %d/%d: loss=%.4f  policy=%.4f  value=%.4f",
                round_idx, step, config.T_steps,
                loss.item(), policy_loss.item(), value_loss.item(),
            )

    n = max(steps_done, 1)
    return {
        "mean_loss":   total_loss / n,
        "mean_policy": total_policy / n,
        "mean_value":  total_value / n,
        "steps":       steps_done,
    }


def outer_loop(config: TrainingConfig) -> AlphaRewriteNet:
    """Run the full AlphaZero outer loop.

    Parameters
    ----------
    config : TrainingConfig
        All hyperparameters.

    Returns
    -------
    AlphaRewriteNet
        The trained network after K rounds.
    """
    from data_generation.egraph_server import EGraphServer
    from data_generation.mcts import MCTSServer
    from data_generation.replay_buffer import ReplayBuffer
    from data_generation.self_play import InferenceServer, run_self_play

    # ---- Initialise components ----------------------------------------
    net = initialize_parameters(config)
    if config.resume_from is not None:
        from training.checkpoint import load_checkpoint
        load_checkpoint(config.resume_from, net, device=config.device)

    optimizer = optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=0.0  # L2 via loss.py
    )

    egraph_server   = EGraphServer()
    mcts_server     = MCTSServer(
        egraph_server=egraph_server,
        c_puct=config.c_puct,
        num_simulations=config.num_simulations,
    )
    inference_server = InferenceServer(
        net=net,
        egraph_server=egraph_server,
        device=config.device,
    )
    replay_buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        cost_metric=config.cost_metric,
    )

    # ---- Publish initial weights ----------------------------------------
    publish_to_inference_server(inference_server, net)

    # ---- Outer loop: round k = 0..K-1 -----------------------------------
    for k in range(config.K_rounds):
        logger.info("=== Training round %d / %d ===", k, config.K_rounds - 1)

        # Phase 1: Data Generation via self-play.
        run_self_play(
            egraph_server=egraph_server,
            mcts_server=mcts_server,
            inference_server=inference_server,
            terms=config.training_terms,
            n_games=config.n_games,
            budget=config.budget,
            replay_buffer=replay_buffer,
            temperature=config.temperature,
        )
        logger.info("Replay buffer size after round %d: %d", k, len(replay_buffer))

        # Phase 2: Gradient training.
        stats = _train_one_round(net, replay_buffer, optimizer, config, round_idx=k)
        logger.info("Round %d training done: %s", k, stats)

        # Phase 3: Publish updated weights.
        publish_to_inference_server(inference_server, net)

        # Checkpoint.
        if (k + 1) % config.checkpoint_every == 0 or k == config.K_rounds - 1:
            ckpt_path = config.checkpoint_dir / f"theta_k{k:04d}.pt"
            save_checkpoint(ckpt_path, net, round_idx=k, step=(k + 1) * config.T_steps, config=config)

    logger.info("Training complete after %d rounds.", config.K_rounds)
    return net

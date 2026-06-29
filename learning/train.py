"""
Entry point: train a walking policy with Brax PPO on the MJX env.

This is the "logic" wiring layer: it pulls the functional env together with the
PPO algorithm and the hyperparameters from the config, runs training, and saves
the policy parameters.  The algorithm is isolated here so it can be swapped
(another trainer) without touching the env or components.

    python -m learning.train

Throughput note: PPO defaults assume thousands of parallel envs.  On a CPU-only
Mac this runs but is slow — drop ppo.num_envs / num_timesteps in the config for a
quick smoke test, and do real runs on a CUDA machine.
"""

from __future__ import annotations

import functools
import os
import time

import jax
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from learning.config import (
    Config, EXPERIMENTS, checkpoint_path, default_config, get_config, smoke_ify,
)
from learning.env.mjx_env import make_env


def train(cfg: Config = None, params_path: str = None):
    cfg = cfg or default_config()
    params_path = params_path or checkpoint_path(cfg.name)
    p = cfg.ppo

    env = make_env(cfg)
    eval_env = make_env(cfg)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=p.policy_hidden,
        value_hidden_layer_sizes=p.value_hidden,
    )

    t0 = time.time()

    def _reward_key(metrics):
        """Find brax's eval-return metric (the exact name varies by version)."""
        for key in ("eval/episode_reward", "eval/episode_return"):
            if key in metrics:
                return key
        cands = [k for k in metrics
                 if "reward" in k and "std" not in k and k.startswith("eval")]
        return cands[0] if cands else None

    def progress(num_steps, metrics):
        if num_steps == 0:
            print("eval metric keys:", sorted(k for k in metrics if "eval" in k))
        key = _reward_key(metrics)
        reward = float(metrics[key]) if key else float("nan")
        length = float(metrics.get("eval/avg_episode_length", float("nan")))
        print(f"[{time.time()-t0:6.0f}s] steps={num_steps:>12,}  "
              f"eval_reward={reward:8.3f}  ep_len={length:7.1f}  ({key})")

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=p.num_timesteps,
        num_envs=p.num_envs,
        episode_length=p.episode_length,
        batch_size=p.batch_size,
        num_minibatches=p.num_minibatches,
        num_updates_per_batch=p.num_updates_per_batch,
        unroll_length=p.unroll_length,
        learning_rate=p.learning_rate,
        entropy_cost=p.entropy_cost,
        discounting=p.discounting,
        reward_scaling=p.reward_scaling,
        normalize_observations=p.normalize_observations,
        num_evals=p.num_evals,
        seed=p.seed,
        network_factory=network_factory,
        progress_fn=progress,
    )

    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    model.save_params(params_path, params)
    print(f"\nsaved policy -> {params_path}")
    return make_inference_fn, params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline", choices=sorted(EXPERIMENTS),
                        help="experiment config to train (see learning/config.py)")
    parser.add_argument("--smoke", action="store_true",
                        help="short run to verify the training loop end-to-end")
    parser.add_argument("--steps", type=int, default=None,
                        help="override num_timesteps (e.g. 30000000)")
    parser.add_argument("--envs", type=int, default=None,
                        help="override num_envs (lower on CPU, higher on GPU)")
    args = parser.parse_args()

    cfg = get_config(args.config)
    if args.smoke:
        cfg = smoke_ify(cfg)
    if args.steps is not None:
        cfg.ppo.num_timesteps = args.steps
    if args.envs is not None:
        cfg.ppo.num_envs = args.envs

    train(cfg=cfg)

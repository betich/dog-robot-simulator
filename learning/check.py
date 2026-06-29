"""
Entry point: validate the environment wiring (run this FIRST).

Builds the env from the config, then does a short random rollout and prints the
shapes and values flowing through it.  No training, no neural net — this just
confirms the model loads, the components compose, and reset/step are jit-able.

    python -m learning.check
"""

from __future__ import annotations

import jax

from learning.config import default_config
from learning.env.mjx_env import make_env
from learning.registry import (ACTIONS, COMMANDS, OBSERVATIONS, REWARDS,
                               TERMINATIONS)


def main():
    cfg = default_config()

    print("registered components")
    print(f"  observations: {OBSERVATIONS.names()}")
    print(f"  rewards:      {REWARDS.names()}")
    print(f"  terminations: {TERMINATIONS.names()}")
    print(f"  actions:      {ACTIONS.names()}")
    print(f"  commands:     {COMMANDS.names()}")

    env = make_env(cfg)
    print(f"\nenv built — obs_size={env.observation_size}  "
          f"action_size={env.action_size}  "
          f"sim_dt={env._sim_dt*1e3:.1f}ms  ctrl_dt={env.dt*1e3:.1f}ms  "
          f"substeps={env.n_substeps}")

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    print(f"\nreset ok — obs{tuple(state.obs.shape)}  command={state.info['command']}")

    print("\nrandom rollout:")
    for i in range(8):
        rng, key = jax.random.split(rng)
        action = jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)
        state = step(state, action)
        terms = {k.split("/")[-1]: round(float(v), 3)
                 for k, v in state.metrics.items()}
        print(f"  step {i}: reward={float(state.reward):+.4f}  "
              f"done={float(state.done):.0f}  terms={terms}")

    print("\nOK — environment is wired correctly.")


if __name__ == "__main__":
    main()

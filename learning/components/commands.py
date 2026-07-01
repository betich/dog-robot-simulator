"""
Command samplers (functional, interchangeable).

A command is the task the policy must follow.  Each sampler is a pure function
(rng, cfg) -> command vector, drawn fresh at episode reset and (optionally)
resampled mid-episode by the env.  Swap the task by registering a new sampler.
"""

from __future__ import annotations

import jax
import jax.numpy as jp

from learning.registry import COMMANDS


@COMMANDS.register("velocity_2d")
def velocity_2d(rng, cfg) -> jp.ndarray:
    """Uniformly sample (vx, vy, wz) from the ranges in config.command."""
    kx, ky, kw = jax.random.split(rng, 3)
    c = cfg.command
    vx = jax.random.uniform(kx, (), minval=c.lin_vel_x[0], maxval=c.lin_vel_x[1])
    vy = jax.random.uniform(ky, (), minval=c.lin_vel_y[0], maxval=c.lin_vel_y[1])
    wz = jax.random.uniform(kw, (), minval=c.ang_vel_z[0], maxval=c.ang_vel_z[1])
    return jp.array([vx, vy, wz])


@COMMANDS.register("velocity_modes")
def velocity_modes(rng, cfg) -> jp.ndarray:
    """Mix uniform draws with canonical full-magnitude commands.

    Uniform sampling (velocity_2d) puts almost no density on the *extremes* of the
    box, so a policy is never pressured to learn the full locomotion loop — fast
    forward, backward, strafe, turn — and settles on a near-stationary tracker (see
    results/README.md).  This sampler instead draws, with probability
    `cfg.command.mode_prob`, one of the loop's canonical commands at full
    magnitude (each axis pushed to a range endpoint, plus a stop), so every loop
    behaviour is seen often.  The rest of the time it falls back to a uniform draw
    for interpolation coverage.  Pure JAX (runs under reset/step's jit+vmap)."""
    c = cfg.command
    k_pick, k_mode, k_jit, k_uni = jax.random.split(rng, 4)

    # The loop's canonical full-magnitude commands (concrete values from config).
    modes = jp.array([
        [c.lin_vel_x[1], 0.0,            0.0           ],  # forward / run (max vx)
        [c.lin_vel_x[0], 0.0,            0.0           ],  # backward (min vx)
        [0.0,            c.lin_vel_y[1], 0.0           ],  # strafe left (max vy)
        [0.0,            c.lin_vel_y[0], 0.0           ],  # strafe right (min vy)
        [0.0,            0.0,            c.ang_vel_z[1]],  # turn left (max wz)
        [0.0,            0.0,            c.ang_vel_z[0]],  # turn right (min wz)
        [0.0,            0.0,            0.0           ],  # stop
    ])
    idx = jax.random.randint(k_mode, (), 0, modes.shape[0])
    mode_cmd = modes[idx]
    # Jitter the non-stop axes a little so the policy doesn't overfit exact values.
    jitter = jax.random.uniform(k_jit, (3,),
                                minval=-c.mode_jitter, maxval=c.mode_jitter)
    mode_cmd = jp.where(mode_cmd != 0.0, mode_cmd + jitter, mode_cmd)

    uniform_cmd = velocity_2d(k_uni, cfg)
    take_mode = jax.random.bernoulli(k_pick, c.mode_prob)
    return jp.where(take_mode, mode_cmd, uniform_cmd)

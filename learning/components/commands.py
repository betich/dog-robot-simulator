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

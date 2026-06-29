"""
Termination terms (functional, interchangeable).

Each term is a pure function (env, data, info) -> bool scalar.  The env OR-s the
terms listed in config.termination.terms; if any is true the episode ends (and
the trainer auto-resets).  Keep these to genuine failure conditions — time limits
are handled separately by the trainer's episode_length.
"""

from __future__ import annotations

import jax.numpy as jp

from learning.registry import TERMINATIONS


@TERMINATIONS.register("fell_over")
def fell_over(env, data, info) -> jp.ndarray:
    """True if the torso has collapsed (too low) or toppled (too tilted)."""
    height = env.base_pos(data)[2]
    tilt = jp.linalg.norm(env.projected_gravity(data)[:2])
    too_low = height < env.cfg.termination.min_base_height
    too_tilted = tilt > env.cfg.termination.max_tilt
    return jp.logical_or(too_low, too_tilted)

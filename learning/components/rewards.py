"""
Reward terms (functional, interchangeable).

Each term is a pure function (env, data, action, info) -> scalar.  The env
computes  sum(weight * term)  over the terms listed in config.reward.weights and
multiplies by ctrl_dt so the total is timestep-invariant.  Sign convention: write
the term as a positive quantity and let the *weight* decide encourage (+) vs
penalize (-).  Tune behaviour by editing weights in the config, not here.
"""

from __future__ import annotations

import jax.numpy as jp

from learning.registry import REWARDS


# ── task: track the commanded velocity ───────────────────────────────────────
@REWARDS.register("track_lin_vel")
def track_lin_vel(env, data, action, info) -> jp.ndarray:
    """Exp kernel on planar linear-velocity tracking error (1 = perfect)."""
    cmd = info["command"]
    vel = env.base_lin_vel_body(data)
    err = jp.sum(jp.square(cmd[:2] - vel[:2]))
    return jp.exp(-err / env.cfg.reward.tracking_sigma)


@REWARDS.register("track_ang_vel")
def track_ang_vel(env, data, action, info) -> jp.ndarray:
    """Exp kernel on yaw-rate tracking error (1 = perfect)."""
    cmd = info["command"]
    wz = env.base_ang_vel_body(data)[2]
    err = jp.square(cmd[2] - wz)
    return jp.exp(-err / env.cfg.reward.tracking_sigma)


@REWARDS.register("alive")
def alive(env, data, action, info) -> jp.ndarray:
    """Constant bonus for not having terminated this step."""
    return jp.array(1.0)


@REWARDS.register("base_height")
def base_height(env, data, action, info) -> jp.ndarray:
    """Squared torso-height deviation from target (use with a NEGATIVE weight).

    A penalty, not a bonus: it is 0 when the robot stands at target height and
    grows as it crouches, so it discourages the belly-down shuffle WITHOUT paying
    the robot to stand still.  (A positive 'tall' bonus competes with the walking
    reward and creates a march-in-place optimum — which is exactly what happened.)
    """
    h = env.base_pos(data)[2]
    return jp.square(h - env.cfg.reward.base_height_target)


# ── shaping penalties (use with negative weights) ────────────────────────────
@REWARDS.register("lin_vel_z")
def lin_vel_z(env, data, action, info) -> jp.ndarray:
    """Penalize bouncing (vertical velocity)."""
    return jp.square(env.base_lin_vel_body(data)[2])


@REWARDS.register("ang_vel_xy")
def ang_vel_xy(env, data, action, info) -> jp.ndarray:
    """Penalize roll/pitch rates."""
    return jp.sum(jp.square(env.base_ang_vel_body(data)[:2]))


@REWARDS.register("orientation")
def orientation(env, data, action, info) -> jp.ndarray:
    """Penalize non-flat torso (projected gravity should be [0,0,-1])."""
    return jp.sum(jp.square(env.projected_gravity(data)[:2]))


@REWARDS.register("torques")
def torques(env, data, action, info) -> jp.ndarray:
    """Penalize actuator effort (energy / smoothness)."""
    return jp.sum(jp.square(env.torques(data)))


@REWARDS.register("action_rate")
def action_rate(env, data, action, info) -> jp.ndarray:
    """Penalize jerky changes between consecutive actions."""
    return jp.sum(jp.square(action - info["last_action"]))


@REWARDS.register("joint_limits")
def joint_limits(env, data, action, info) -> jp.ndarray:
    """Penalize joints driven outside their allowed range."""
    q = env.joint_pos(data)
    lower, upper = env.joint_range[:, 0], env.joint_range[:, 1]
    over = jp.maximum(q - upper, 0.0) + jp.maximum(lower - q, 0.0)
    return jp.sum(over)

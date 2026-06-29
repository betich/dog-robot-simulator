"""
Observation terms (functional, interchangeable).

Each term is a pure function (env, data, info) -> 1-D array.  The env concatenates
the terms listed in config.obs.terms, in order, to build the policy input.  Add a
new sensor by writing a function here and listing its name in the config.
"""

from __future__ import annotations

import jax.numpy as jp

from learning.registry import OBSERVATIONS


@OBSERVATIONS.register("base_lin_vel")
def base_lin_vel(env, data, info) -> jp.ndarray:
    """Torso linear velocity in the body frame (m/s)."""
    return env.base_lin_vel_body(data)


@OBSERVATIONS.register("base_ang_vel")
def base_ang_vel(env, data, info) -> jp.ndarray:
    """Torso angular velocity in the body frame (rad/s)."""
    return env.base_ang_vel_body(data)


@OBSERVATIONS.register("projected_gravity")
def projected_gravity(env, data, info) -> jp.ndarray:
    """Gravity direction in the body frame; encodes roll/pitch (upright -> [0,0,-1])."""
    return env.projected_gravity(data)


@OBSERVATIONS.register("command")
def command(env, data, info) -> jp.ndarray:
    """The (vx, vy, wz) velocity command the policy is asked to track."""
    return info["command"]


@OBSERVATIONS.register("joint_pos_rel")
def joint_pos_rel(env, data, info) -> jp.ndarray:
    """Joint angles relative to the nominal standing pose (rad)."""
    return env.joint_pos(data) - env.default_pose


@OBSERVATIONS.register("joint_vel")
def joint_vel(env, data, info) -> jp.ndarray:
    """Joint velocities (rad/s)."""
    return env.joint_vel(data)


@OBSERVATIONS.register("last_action")
def last_action(env, data, info) -> jp.ndarray:
    """Previous policy action (helps the policy produce smooth motion)."""
    return info["last_action"]

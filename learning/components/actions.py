"""
Action mappings (functional, interchangeable).

Each mapping is a pure function (env, action, data, info) -> ctrl vector (nu,).
The policy always outputs `action` in roughly [-1, 1]; the mapping decides what
that means physically.  This is the single seam between "what the policy decides"
and "what the actuators receive", so it is the natural place to inject a gait
prior later (e.g. residual-on-trot).  Choose one with config.action.name.
"""

from __future__ import annotations

import jax.numpy as jp

from learning.gait_prior import phase_for_step, trot_delta
from learning.registry import ACTIONS


@ACTIONS.register("residual_pose")
def residual_pose(env, action, data, info) -> jp.ndarray:
    """
    ctrl = nominal standing pose + scale * action  (position targets).

    Simple and effective: the policy only has to learn *offsets* from a stance
    that already holds the robot up, so early training rarely falls over.
    """
    ctrl = env.default_pose + env.cfg.action.scale * action
    return jp.clip(ctrl, env.ctrl_range[:, 0], env.ctrl_range[:, 1])


@ACTIONS.register("residual_gait")
def residual_gait(env, action, data, info) -> jp.ndarray:
    """
    ctrl = scripted_trot(command, phase) + scale * action  (position targets).

    Hands the policy a *walking* prior: the scripted trot from sim/gait_controller
    already lifts feet and steps in the commanded direction, so the policy only
    learns corrections to it instead of discovering locomotion from scratch. The
    phase is derived from the env step counter, so no extra env state is needed.
    """
    phase = phase_for_step(info["step"], env.dt)
    base = env.default_pose + trot_delta(info["command"], phase)
    ctrl = base + env.cfg.action.scale * action
    return jp.clip(ctrl, env.ctrl_range[:, 0], env.ctrl_range[:, 1])


@ACTIONS.register("direct_pose")
def direct_pose(env, action, data, info) -> jp.ndarray:
    """
    ctrl = scale * action, interpreted as absolute position targets around the
    nominal pose with no per-joint range remap.  More general, less stable early.
    """
    ctrl = env.default_pose + env.cfg.action.scale * action
    # identical math to residual_pose today; kept distinct so the two seams can
    # diverge (e.g. direct_pose could ignore the nominal pose entirely).
    return jp.clip(ctrl, env.ctrl_range[:, 0], env.ctrl_range[:, 1])


@ACTIONS.register("torque")
def torque(env, action, data, info) -> jp.ndarray:
    """
    ctrl = scale * action interpreted as joint torque.

    NOTE: the shipped MJCF uses *position* actuators, so this only makes physical
    sense after switching the <actuator> entries to <motor>.  Provided as a seam
    for when you mature toward torque-control RL.
    """
    return jp.clip(env.cfg.action.scale * action,
                   env.ctrl_range[:, 0], env.ctrl_range[:, 1])

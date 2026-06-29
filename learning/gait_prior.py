"""
JAX port of the scripted trot from sim/gait_controller.py — the *prior* for the
`residual_gait` action mapping.

The scripted controller is stateful python (it advances a phase each call); for
RL we need a pure function of (command, phase) that jit/vmaps across thousands of
envs.  The walking *numbers* (link lengths, gait params, per-leg sign tables) are
imported from sim.gait_controller so there is one source of truth — only the
control flow is reimplemented in jax here.

Output is a (12,) joint *delta* in MuJoCo actuator order
(LF, RF, LH, RH) x (HAA, HFE, KFE), to be added to the standing pose.  HAA is
left at nominal (delta 0); only HFE/KFE move, exactly as in the scripted trot.
"""

from __future__ import annotations

import jax.numpy as jp

from sim import gait_controller as gc

_LEGS = gc.LEGS                          # ['LF', 'RF', 'LH', 'RH'] = actuator leg order

# Per-leg constants, arranged to match _LEGS (and thus the 1::3 / 2::3 slices below).
_OFFS    = jp.array([gc.PHASE_OFFSET[l] for l in _LEGS])
_SIDE    = jp.array([gc._SIDE[l]        for l in _LEGS])
_HFE_DIR = jp.array([gc.HFE_DIR[l]      for l in _LEGS])
_KFE_DIR = jp.array([gc.KFE_DIR[l]      for l in _LEGS])

_L1, _L2     = gc.L_THIGH, gc.L_SHANK
_PHI0, _PSI0 = gc._PHI0, gc._PSI0
_DEPTH0      = gc._DEPTH0
_DUTY, _LIFT = gc.DUTY_CYCLE, gc.SWING_LIFT
_FREQ        = gc.GAIT_FREQ
_STRIDE_PER_V, _STRIDE_MAX, _TURN = gc.STRIDE_PER_V, gc.STRIDE_MAX, gc.TURN_STRIDE

_RMIN = abs(_L1 - _L2) + 1e-4
_RMAX = _L1 + _L2 - 1e-4


def phase_for_step(step, dt: float):
    """Gait phase in [0,1), derived from the env step counter (no extra state)."""
    return (step.astype(jp.float32) * dt * _FREQ) % 1.0


def trot_delta(command: jp.ndarray, phase: jp.ndarray) -> jp.ndarray:
    """Joint deltas (12,) for the scripted trot at this command and phase."""
    vx, wz = command[0], command[2]
    fwd = jp.clip(vx * _STRIDE_PER_V, -_STRIDE_MAX, _STRIDE_MAX)

    tau = (phase + _OFFS) % 1.0                      # (4,) per-leg phase
    stride = fwd - _SIDE * wz * _TURN                # skid-steer yaw bias

    # foot (x forward, depth below hip) — stance sweeps back, swing lifts & returns
    in_stance = tau < _DUTY
    s = tau / _DUTY
    u = (tau - _DUTY) / (1.0 - _DUTY)
    x = jp.where(in_stance, stride * (0.5 - s), stride * (u - 0.5))
    depth = jp.where(in_stance, _DEPTH0, _DEPTH0 - _LIFT * jp.sin(jp.pi * u))

    # planar 2-link IK -> (phi hip pitch, psi knee flexion), vectorized over legs
    r = jp.clip(jp.hypot(x, depth), _RMIN, _RMAX)
    cos_knee = jp.clip((_L1**2 + _L2**2 - r**2) / (2 * _L1 * _L2), -1.0, 1.0)
    psi = jp.pi - jp.arccos(cos_knee)
    cos_g = jp.clip((_L1**2 + r**2 - _L2**2) / (2 * _L1 * r), -1.0, 1.0)
    phi = jp.arctan2(x, depth) + jp.arccos(cos_g)

    dphi = _HFE_DIR * (phi - _PHI0)                  # (4,) HFE deltas
    dpsi = _KFE_DIR * (psi - _PSI0)                  # (4,) KFE deltas

    delta = jp.zeros(12)
    delta = delta.at[1::3].set(dphi)                 # HFE = actuator idx 1,4,7,10
    delta = delta.at[2::3].set(dpsi)                 # KFE = actuator idx 2,5,8,11
    return delta

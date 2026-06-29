"""
Small jax quaternion helpers (MuJoCo quaternion order: w, x, y, z).

Kept dependency-light and pure so observation/reward components can use them
without pulling in brax.math.
"""

from __future__ import annotations

import jax.numpy as jp


def quat_conj(q: jp.ndarray) -> jp.ndarray:
    """Conjugate (inverse for a unit quaternion)."""
    w, x, y, z = q
    return jp.array([w, -x, -y, -z])


def rotate(q: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
    """Rotate vector v from the local frame of q into the world frame."""
    w, x, y, z = q
    # v + 2 * cross(q_xyz, cross(q_xyz, v) + w*v)
    u = jp.array([x, y, z])
    t = 2.0 * jp.cross(u, v)
    return v + w * t + jp.cross(u, t)


def inv_rotate(q: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
    """Rotate vector v from world frame into the local frame of q."""
    return rotate(quat_conj(q), v)


def projected_gravity(q: jp.ndarray) -> jp.ndarray:
    """World 'down' expressed in the body frame; flat & upright -> [0,0,-1]."""
    return inv_rotate(q, jp.array([0.0, 0.0, -1.0]))

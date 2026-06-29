"""
Robot-specific facts for ANYmal C.

The generic environment (env/mjx_env.py) is robot-agnostic: everything it needs
to know about *this particular robot* — where the model file is, the joint order,
the nominal standing pose, the control ranges, which body is the torso — comes
from the `RobotInfo` this module builds.  Swapping to a different quadruped later
means writing another loader, not touching the env.

This is the only place on the RL side that touches the raw `mujoco.MjModel`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import mjx

from learning.config import RobotConfig

# Repo root = parent of the learning/ package, so model paths in the config can
# be written relative to the project root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RobotInfo:
    mj_model: "mujoco.MjModel"          # CPU model — for reset constants + viewer
    mjx_model: "mjx.Model"              # device model — for the batched sim
    nominal_qpos: np.ndarray            # (nq,) full standing pose (keyframe)
    default_pose: np.ndarray            # (nu,) standing joint angles only
    ctrl_range: np.ndarray              # (nu, 2) actuator ctrl limits
    joint_range: np.ndarray             # (nu, 2) joint position limits
    torso_body_id: int                  # body id of the base/torso
    nq: int
    nv: int
    nu: int


def load_robot(cfg: RobotConfig) -> RobotInfo:
    path = cfg.model_xml
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"robot model not found: {path}")

    mj_model = mujoco.MjModel.from_xml_path(path)

    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, cfg.keyframe)
    if key_id < 0:
        raise ValueError(
            f"keyframe '{cfg.keyframe}' not found in {path}. "
            f"available keys: {[mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(mj_model.nkey)]}"
        )
    nominal_qpos = np.array(mj_model.key_qpos[key_id], dtype=np.float64)

    # Actuator i drives joint = actuator_trnid[i, 0]; default pose = those qpos.
    default_pose = np.zeros(mj_model.nu, dtype=np.float64)
    joint_range = np.zeros((mj_model.nu, 2), dtype=np.float64)
    for a in range(mj_model.nu):
        jnt = mj_model.actuator_trnid[a, 0]
        default_pose[a] = nominal_qpos[mj_model.jnt_qposadr[jnt]]
        joint_range[a] = mj_model.jnt_range[jnt]

    torso_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")

    return RobotInfo(
        mj_model=mj_model,
        mjx_model=mjx.put_model(mj_model),
        nominal_qpos=nominal_qpos,
        default_pose=default_pose,
        ctrl_range=np.array(mj_model.actuator_ctrlrange, dtype=np.float64),
        joint_range=joint_range,
        torso_body_id=torso_body_id,
        nq=mj_model.nq,
        nv=mj_model.nv,
        nu=mj_model.nu,
    )

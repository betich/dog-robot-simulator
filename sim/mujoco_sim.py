"""
MuJoCo simulation loop for ANYmal C.

Runs at model.opt.timestep (default 2 ms for ANYmal C).
Publishes robot state via ZMQ after every physics step.
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from gait_controller import GaitController
from zmq_bridge import ZMQBridge

JOINT_ORDER = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]


class MuJoCoSim:
    def __init__(self, model_path: str, bridge: ZMQBridge, gait: GaitController):
        self.model  = mujoco.MjModel.from_xml_path(model_path)
        self.data   = mujoco.MjData(self.model)
        self.bridge = bridge
        self.gait   = gait

        # Map actuator name → ctrl index (built from MJCF at load time)
        self._act_idx: dict[str, int] = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._act_idx[name] = i

        missing = [j for j in JOINT_ORDER if j not in self._act_idx]
        if missing:
            print(f"[sim] WARNING: actuators not found in model: {missing}")
            print(f"[sim] available actuators: {list(self._act_idx)}")

        # Map joint name → qpos/qvel offset (joints start after free-joint at idx 7/6)
        self._jnt_qpos_idx: dict[str, int] = {}
        self._jnt_qvel_idx: dict[str, int] = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name in JOINT_ORDER:
                self._jnt_qpos_idx[name] = self.model.jnt_qposadr[i]
                self._jnt_qvel_idx[name] = self.model.jnt_dofadr[i]

        print(f"[sim] model loaded — dt={self.model.opt.timestep*1000:.1f} ms  "
              f"nu={self.model.nu}  nq={self.model.nq}")

    def _extract_state(self) -> dict:
        qpos = self.data.qpos
        qvel = self.data.qvel

        # Free-joint: qpos[0:3]=pos, qpos[3:7]=quat(w,x,y,z)
        x, y, z   = float(qpos[0]), float(qpos[1]), float(qpos[2])
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        yaw = float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))

        vx = float(qvel[0])
        vy = float(qvel[1])
        wz = float(qvel[5])

        joint_pos = {
            n: float(qpos[self._jnt_qpos_idx[n]])
            for n in JOINT_ORDER if n in self._jnt_qpos_idx
        }
        joint_vel = {
            n: float(qvel[self._jnt_qvel_idx[n]])
            for n in JOINT_ORDER if n in self._jnt_qvel_idx
        }

        return {
            "t": float(self.data.time),
            "odom": {"x": x, "y": y, "yaw": yaw, "vx": vx, "vy": vy, "wz": wz},
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def _apply_targets(self, targets: dict):
        for name, angle in targets.items():
            if name in self._act_idx:
                self.data.ctrl[self._act_idx[name]] = angle

    def run(self):
        dt = self.model.opt.timestep
        # Gait control runs every N physics steps to reduce IK overhead
        ctrl_every = max(1, round(0.005 / dt))  # ~5 ms control period

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0
            while viewer.is_running():
                t0 = time.perf_counter()

                if step % ctrl_every == 0:
                    cmd     = self.bridge.recv_cmd()
                    targets = self.gait.step(cmd)
                    self._apply_targets(targets)

                mujoco.mj_step(self.model, self.data)

                if step % ctrl_every == 0:
                    self.bridge.pub_state(self._extract_state())

                viewer.sync()
                step += 1

                # Soft real-time: sleep only if ahead of wall clock
                elapsed   = time.perf_counter() - t0
                remaining = dt - elapsed
                if remaining > 1e-4:
                    time.sleep(remaining)

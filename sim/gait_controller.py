"""
Kinematic IK trot gait controller for ANYmal C.

Joint naming convention
-----------------------
Leg prefix:
    LF  Left  Front
    RF  Right Front
    LH  Left  Hind
    RH  Right Hind

Joint suffix:
    HAA  Hip Abduction / Adduction  — swings the leg in/out laterally
    HFE  Hip Flexion  / Extension   — swings the leg forward/backward
    KFE  Knee Flexion / Extension   — bends/straightens the knee

Full joint list (12 total, 3 per leg):
    LF_HAA  LF_HFE  LF_KFE
    RF_HAA  RF_HFE  RF_KFE
    LH_HAA  LH_HFE  LH_KFE
    RH_HAA  RH_HFE  RH_KFE

All parameters are derived from the anybotics_anymal_c MJCF.
Verify HIP_POS, L_THIGH, L_SHANK against the actual XML before running.
If knees appear inverted in the viewer, flip the sign of KFE_SIGN.
"""

import math
import numpy as np

# ── Kinematic parameters (from anybotics_anymal_c MJCF, approximate) ─────────
HIP_POS = {                         # hip joint position in body frame (m)
    "LF": np.array([ 0.277,  0.116, 0.0]),
    "RF": np.array([ 0.277, -0.116, 0.0]),
    "LH": np.array([-0.277,  0.116, 0.0]),
    "RH": np.array([-0.277, -0.116, 0.0]),
}
HAA_OFFSET = {                      # HAA-to-HFE lateral link length (m)
    "LF":  0.041, "RF": -0.041,
    "LH":  0.041, "RH": -0.041,
}
L_THIGH = 0.35                      # HFE-to-KFE link length (m)
L_SHANK = 0.33                      # KFE-to-foot  link length (m)

# ── Gait parameters ───────────────────────────────────────────────────────────
STAND_HEIGHT = 0.52                 # desired body-to-foot height (m)
SWING_LIFT   = 0.07                 # peak foot clearance during swing (m)
DUTY_CYCLE   = 0.55                 # fraction of cycle in stance
GAIT_FREQ    = 1.5                  # trot frequency (Hz)

# Diagonal pairs share a swing phase in trot
PHASE_OFFSET = {                    # per-leg phase offset (rad)
    "LF": 0.0,       "RH": 0.0,
    "RF": math.pi,   "LH": math.pi,
}

LEGS = ["LF", "RF", "LH", "RH"]


def _solve_ik(leg: str, foot_body: np.ndarray) -> np.ndarray:
    """
    Analytical 3-DOF IK: foot position in body frame → [HAA, HFE, KFE] (rad).

    Coordinate convention (body frame):  x=forward, y=left, z=up.
    KFE sign: positive = knee bends (extension positive means q_kfe > 0 = straight,
    so a crouching stance has q_kfe < 0).  Flip if the model disagrees.
    """
    hip = HIP_POS[leg]
    d   = HAA_OFFSET[leg]       # lateral offset of thigh from HAA axis

    p = foot_body - hip         # foot in hip frame
    px, py, pz = p

    # ── HAA: rotation about body-x through hip ──────────────────────────────
    r_yz = math.sqrt(py**2 + pz**2)
    r_yz = max(r_yz, abs(d) + 1e-6)
    phi_d = math.asin(max(-1.0, min(1.0, d / r_yz)))
    q_haa = math.atan2(py, -pz) - phi_d

    # ── project foot into sagittal plane after HAA rotation ─────────────────
    p_sagx =  px
    p_sagz = -math.sqrt(max(0.0, r_yz**2 - d**2))   # always negative (foot below)

    # ── 2R IK in sagittal plane: HFE and KFE ────────────────────────────────
    r = math.hypot(p_sagx, p_sagz)
    r = max(1e-6, min(r, L_THIGH + L_SHANK - 1e-4))

    cos_c = (r**2 - L_THIGH**2 - L_SHANK**2) / (2 * L_THIGH * L_SHANK)
    cos_c = max(-1.0, min(1.0, cos_c))

    # Negative KFE = knee bent backward (ANYmal convention; tune if needed)
    q_kfe = -math.acos(cos_c)

    gamma = math.atan2(-p_sagz, p_sagx)            # angle from horizontal to foot
    beta  = math.atan2(
        L_SHANK * math.sin(-q_kfe),
        L_THIGH + L_SHANK * math.cos(-q_kfe),
    )
    q_hfe = gamma - beta - math.pi / 2             # from downward vertical

    return np.array([q_haa, q_hfe, q_kfe])


class GaitController:
    def __init__(self, control_freq: float = 200.0):
        self.dt        = 1.0 / control_freq
        self.phase     = 0.0
        self.cycle_len = 1.0 / GAIT_FREQ           # seconds
        # stride length = velocity × stance time
        self._stride_scale = DUTY_CYCLE * self.cycle_len

    # ── nominal foot position for standing still ──────────────────────────────
    def _nominal_foot(self, leg: str) -> np.ndarray:
        h = HIP_POS[leg]
        return np.array([h[0], h[1] + HAA_OFFSET[leg], -STAND_HEIGHT])

    # ── foot position during swing (t ∈ [0, 1]) ──────────────────────────────
    def _swing_foot(self, leg: str, t: float, cmd: dict) -> np.ndarray:
        nom = self._nominal_foot(leg)
        sx  = cmd.get("linear_x", 0.0) * self._stride_scale
        sy  = cmd.get("linear_y", 0.0) * self._stride_scale
        sz  = 0.0
        # interpolate from back (-stride/2) to front (+stride/2)
        return np.array([
            nom[0] + sx * (t - 0.5),
            nom[1] + sy * (t - 0.5),
            nom[2] + SWING_LIFT * math.sin(math.pi * t) + sz,
        ])

    # ── foot position during stance (t ∈ [0, 1]) ─────────────────────────────
    def _stance_foot(self, leg: str, t: float, cmd: dict) -> np.ndarray:
        nom = self._nominal_foot(leg)
        sx  = cmd.get("linear_x", 0.0) * self._stride_scale
        sy  = cmd.get("linear_y", 0.0) * self._stride_scale
        # foot moves backward relative to body as body travels forward
        return np.array([
            nom[0] + sx * (0.5 - t),
            nom[1] + sy * (0.5 - t),
            nom[2],
        ])

    def step(self, cmd: dict) -> dict:
        """Advance gait by one control timestep. Returns {joint_name: angle_rad}."""
        self.phase = (self.phase + 2 * math.pi * GAIT_FREQ * self.dt) % (2 * math.pi)

        targets = {}
        for leg in LEGS:
            leg_phase   = (self.phase + PHASE_OFFSET[leg]) % (2 * math.pi)
            normalized  = leg_phase / (2 * math.pi)    # 0 … 1

            if normalized > DUTY_CYCLE:
                t       = (normalized - DUTY_CYCLE) / (1.0 - DUTY_CYCLE)
                foot    = self._swing_foot(leg, t, cmd)
            else:
                t       = normalized / DUTY_CYCLE
                foot    = self._stance_foot(leg, t, cmd)

            q = _solve_ik(leg, foot)
            targets[f"{leg}_HAA"] = float(q[0])
            targets[f"{leg}_HFE"] = float(q[1])
            targets[f"{leg}_KFE"] = float(q[2])

        return targets

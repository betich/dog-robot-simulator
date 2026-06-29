"""
Kinematic trot gait controller for ANYmal C.

Design
------
This controller is anchored on the *authoritative* standing pose shipped with
the model (the "standing" keyframe in anymal_c_mjx.xml).  That keyframe is the
ground truth for this MJCF's joint-sign conventions:

    base:  z = 0.56, quat = identity
    LF: HAA=0,  HFE=+0.5236, KFE=-0.7854
    RF: HAA=0,  HFE=+0.5236, KFE=-0.7854
    LH: HAA=0,  HFE=-0.5236, KFE=+0.7854   <- signs flipped vs the front legs
    RH: HAA=0,  HFE=-0.5236, KFE=+0.7854

The hind legs use *opposite-signed* HFE/KFE because the MJCF flips their joint
axes (axis="-1 0 0") and thigh quaternions.  A naive analytical IK that returns
one sign convention for all four legs folds the hind legs the wrong way and the
robot tips over — which is exactly what the previous controller did.

Instead of trusting Cartesian IK against the model's (rotated) body frames, we:

  1. Treat each leg as a planar 2-link mechanism in its own sagittal plane.
  2. Generate the foot trajectory (swing/stance) in that plane.
  3. Solve a clean planar 2-link IK for a canonical (hip-pitch, knee-flexion).
  4. Map the *delta from neutral* onto the model joints via a per-leg sign so
     that the neutral foot reproduces the keyframe angles exactly.

Because the standing pose is the keyframe pose, the robot stands in the authors'
stable stance whenever it is stopped, and only oscillates around it while moving.

Sign conventions (derived from the model's forward kinematics)
--------------------------------------------------------------
Canonical leg plane:  +phi (hip pitch) swings the foot FORWARD (+x),
                      +psi (knee flex) raises the foot (+z, shortens the leg).

To move the foot forward / lift it, the model joints must change as:
    HFE_FORWARD_SIGN, KFE_LIFT_SIGN  =  -1 (front legs),  +1 (hind legs)

If a leg walks *backward* or its knee bends the wrong way in the viewer, flip
that leg's entry in HFE_DIR / KFE_DIR below — they are the single tuning point.

Joint naming
------------
Leg prefix:  LF/RF/LH/RH  (Left/Right, Front/Hind)
Joint suffix: HAA (hip abduction), HFE (hip flexion), KFE (knee flexion)
"""

import math

# ── Standing pose (from the "standing" keyframe, ground truth for this MJCF) ──
_HFE_STAND = 0.5235987755982988      # 30 deg
_KFE_STAND = 0.7853981               # 45 deg

NOMINAL = {                          # (HAA, HFE, KFE) standing angles per leg
    "LF": (0.0,  _HFE_STAND, -_KFE_STAND),
    "RF": (0.0,  _HFE_STAND, -_KFE_STAND),
    "LH": (0.0, -_HFE_STAND,  _KFE_STAND),
    "RH": (0.0, -_HFE_STAND,  _KFE_STAND),
}

# Per-leg mapping from canonical (foot-forward / foot-up) deltas to model joints.
# Front legs need negative deltas, hind legs positive — see module docstring.
HFE_DIR = {"LF": -1.0, "RF": -1.0, "LH": +1.0, "RH": +1.0}
KFE_DIR = {"LF": -1.0, "RF": -1.0, "LH": +1.0, "RH": +1.0}

# ── Leg geometry (effective link lengths in the sagittal plane, metres) ──────
L_THIGH = 0.30                       # HFE → KFE
L_SHANK = 0.33                       # KFE → foot

# Canonical neutral knee flexion (matches the keyframe's 45 deg bend)
_PSI0 = _KFE_STAND
# Canonical neutral hip pitch that places the foot directly below the hip
_PHI0 = math.atan2(L_SHANK * math.sin(_PSI0), L_THIGH + L_SHANK * math.cos(_PSI0))

# ── Gait parameters ──────────────────────────────────────────────────────────
GAIT_FREQ    = 1.25                  # trot cycles per second (Hz)
DUTY_CYCLE   = 0.6                   # fraction of the cycle a foot is in stance
SWING_LIFT   = 0.10                  # peak foot clearance during swing (m)
STRIDE_PER_V = 0.75                  # foot stride (m) per (m/s) of body velocity
STRIDE_MAX   = 0.40                  # cap on half-stride amplitude (m)
TURN_STRIDE  = 0.08                  # per-side stride bias (m) per (rad/s) of yaw

# Trot: diagonal pairs (LF+RH) and (RF+LH) move together, half a cycle apart.
PHASE_OFFSET = {"LF": 0.0, "RH": 0.0, "RF": 0.5, "LH": 0.5}

LEGS = ["LF", "RF", "LH", "RH"]

# Left/right sign used to convert a yaw command into a skid-steer stride bias.
_SIDE = {"LF": +1.0, "LH": +1.0, "RF": -1.0, "RH": -1.0}   # +1 = left side


def _planar_ik(x: float, depth: float) -> tuple[float, float]:
    """
    Canonical planar 2-link IK.

    Args:
        x:     foot offset forward of the hip (m), +forward.
        depth: foot distance below the hip (m), +down.

    Returns:
        (phi, psi) — hip pitch from straight-down (+forward) and knee flexion
        (>= 0, 0 = straight leg).
    """
    r = math.hypot(x, depth)
    r = max(abs(L_THIGH - L_SHANK) + 1e-4, min(r, L_THIGH + L_SHANK - 1e-4))

    # knee flexion: 0 when the leg is straight (r = L1 + L2)
    cos_knee = (L_THIGH**2 + L_SHANK**2 - r**2) / (2 * L_THIGH * L_SHANK)
    cos_knee = max(-1.0, min(1.0, cos_knee))
    psi = math.pi - math.acos(cos_knee)

    # hip pitch: direction to the foot, plus the thigh's lead over that line
    alpha = math.atan2(x, depth)
    cos_g = (L_THIGH**2 + r**2 - L_SHANK**2) / (2 * L_THIGH * r)
    cos_g = max(-1.0, min(1.0, cos_g))
    gamma = math.acos(cos_g)
    phi = alpha + gamma
    return phi, psi


# Neutral foot depth below the hip (canonical), used as the stance reference.
_DEPTH0 = (L_THIGH * math.cos(_PHI0)
           + L_SHANK * math.cos(_PHI0 - _PSI0))


class GaitController:
    def __init__(self, control_freq: float = 200.0):
        self.dt    = 1.0 / control_freq
        self.phase = 0.0               # global gait phase, 0 … 1

    # ── public: the standing pose, for sim initialisation / "stopped" hold ────
    def nominal_targets(self) -> dict:
        targets = {}
        for leg in LEGS:
            haa, hfe, kfe = NOMINAL[leg]
            targets[f"{leg}_HAA"] = haa
            targets[f"{leg}_HFE"] = hfe
            targets[f"{leg}_KFE"] = kfe
        return targets

    # ── foot (x, depth) in the canonical leg plane for a given leg phase ──────
    def _foot_xz(self, tau: float, stride: float) -> tuple[float, float]:
        """tau in [0,1): stance for tau < DUTY, swing afterwards."""
        if tau < DUTY_CYCLE:
            # STANCE: foot planted, sweeps backward to push the body forward
            s = tau / DUTY_CYCLE                       # 0 → 1
            x = stride * (0.5 - s)                     # +stride/2 → -stride/2
            depth = _DEPTH0
        else:
            # SWING: foot lifts and returns forward
            u = (tau - DUTY_CYCLE) / (1.0 - DUTY_CYCLE)  # 0 → 1
            x = stride * (u - 0.5)                      # -stride/2 → +stride/2
            depth = _DEPTH0 - SWING_LIFT * math.sin(math.pi * u)
        return x, depth

    def step(self, cmd: dict) -> dict:
        """Advance the gait by one control step. Returns {joint_name: angle_rad}."""
        vx   = cmd.get("linear_x", 0.0)
        turn = cmd.get("angular_z", 0.0)

        moving = abs(vx) > 1e-3 or abs(turn) > 1e-3
        if not moving:
            # Hold the authoritative standing pose; freeze the phase.
            self.phase = 0.0
            return self.nominal_targets()

        self.phase = (self.phase + GAIT_FREQ * self.dt) % 1.0

        fwd_stride = max(-STRIDE_MAX, min(STRIDE_MAX, vx * STRIDE_PER_V))

        targets = {}
        for leg in LEGS:
            tau = (self.phase + PHASE_OFFSET[leg]) % 1.0

            # skid-steer: yaw biases each side's stride (outer side strides more)
            stride = fwd_stride - _SIDE[leg] * turn * TURN_STRIDE

            x, depth = self._foot_xz(tau, stride)
            phi, psi = _planar_ik(x, depth)

            haa_n, hfe_n, kfe_n = NOMINAL[leg]
            targets[f"{leg}_HAA"] = haa_n
            targets[f"{leg}_HFE"] = hfe_n + HFE_DIR[leg] * (phi - _PHI0)
            targets[f"{leg}_KFE"] = kfe_n + KFE_DIR[leg] * (psi - _PSI0)

        return targets

"""
Configuration — the single dial board.

This is the one place you choose *which* interchangeable parts to use and set
every hyperparameter.  Nothing here imports jax or mujoco, so it stays cheap to
read and easy to diff.  The environment and trainer receive a `Config` and look
up the named components in the registry.

To swap a part: change a name string (e.g. action.name) or a weight, or write a
new component in learning/components/ and reference it here.  To run an
experiment: copy `default_config()`, tweak, pass it to the entry point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# learning/config.py -> repo root is two levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def checkpoint_path(name: str) -> str:
    """Where train.py saves / play.py loads an experiment's policy params."""
    return os.path.join(REPO_ROOT, "learning", "checkpoints", name, "anymal_ppo")


def results_dir(name: str) -> str:
    """Standard home for an experiment's rendering + metrics (rollout.gif, metrics.txt)."""
    return os.path.join(REPO_ROOT, "results", name)


@dataclass
class RobotConfig:
    # Path is resolved relative to the repo root by robot.py.
    model_xml: str = "mujoco_menagerie/anybotics_anymal_c/scene_mjx.xml"
    keyframe: str = "standing"          # nominal stance to start/anchor from


@dataclass
class SimConfig:
    ctrl_dt: float = 0.02               # policy/control step (s) -> 50 Hz
    # sim_dt comes from the MJCF (model.opt.timestep); n_substeps = ctrl_dt/sim_dt.


@dataclass
class ActionConfig:
    name: str = "residual_gait"         # see components/actions.py for options
    scale: float = 0.3                  # rad; how far the policy can move a joint
    # "residual_gait": ctrl = scripted_trot(cmd, phase) + scale*action  (walking prior)
    # "residual_pose": ctrl = nominal_pose + scale * action  (no gait prior)
    # "direct_pose":   ctrl = scale * action mapped into ctrlrange
    # "torque":        ctrl = scale * action as joint torque (needs motor actuators)


@dataclass
class ObservationConfig:
    # Concatenated in this order to form the policy input.
    terms: List[str] = field(default_factory=lambda: [
        "base_lin_vel",      # 3  body-frame linear velocity
        "base_ang_vel",      # 3  body-frame angular velocity
        "projected_gravity", # 3  gravity in body frame (encodes tilt)
        "command",           # 3  (vx, vy, wz) the policy must track
        "joint_pos_rel",     # 12 joint angle - nominal
        "joint_vel",         # 12 joint velocity
        "last_action",       # 12 previous action
    ])
    noise_scale: float = 0.0            # >0 adds obs noise (sim2real; off by default)


@dataclass
class RewardConfig:
    # term name -> weight.  The env computes sum(weight * term) * ctrl_dt.
    # Positive = encourage, negative = penalize.  This recipe is the standard
    # "track a velocity command while staying upright and smooth".
    weights: Dict[str, float] = field(default_factory=lambda: {
        "track_lin_vel":   2.0,         # dominant: reaching commanded speed must pay most
        "track_ang_vel":   0.8,
        "alive":           0.5,
        "base_height":    -15.0,        # PENALTY for crouching (0 when tall) — see rewards.py
        "lin_vel_z":      -2.0,
        "ang_vel_xy":     -0.05,
        "orientation":    -5.0,
        "torques":        -0.0001,      # lowered: don't price walking out of the budget
        "action_rate":    -0.01,
        "joint_limits":   -1.0,
    })
    tracking_sigma: float = 0.2         # tighter -> more pressure to hit the full speed
    base_height_target: float = 0.52    # m; allow a slight dip below the 0.56 stance


@dataclass
class TerminationConfig:
    terms: List[str] = field(default_factory=lambda: ["fell_over"])
    min_base_height: float = 0.25       # m; below this = collapsed
    max_tilt: float = 0.7               # |projected gravity xy| above this = toppled


@dataclass
class CommandConfig:
    name: str = "velocity_2d"
    lin_vel_x: Tuple[float, float] = (-0.5, 1.0)   # m/s range sampled per episode
    lin_vel_y: Tuple[float, float] = (-0.4, 0.4)
    ang_vel_z: Tuple[float, float] = (-0.8, 0.8)   # rad/s
    resample_time: float = 5.0          # s; resample command mid-episode (0 = never)


@dataclass
class ResetConfig:
    qpos_noise: float = 0.05            # rad/m jitter on the start pose
    qvel_noise: float = 0.05            # initial velocity jitter


@dataclass
class PPOConfig:
    num_timesteps: int = 60_000_000
    num_envs: int = 4096
    episode_length: int = 1000          # steps (1000 * ctrl_dt = 20 s)
    batch_size: int = 1024
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    unroll_length: int = 20
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.97
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    policy_hidden: Tuple[int, ...] = (128, 128, 128)
    value_hidden: Tuple[int, ...] = (256, 256)
    seed: int = 0
    num_evals: int = 20


@dataclass
class Config:
    name: str = "baseline"              # experiment id; drives checkpoint/results paths
    robot: RobotConfig = field(default_factory=RobotConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def default_config() -> Config:
    """The clean, simple-but-effective starting point (== the 'baseline' experiment)."""
    return Config(name="baseline")


# ── experiments ───────────────────────────────────────────────────────────────
# Each builds on the baseline and changes ONE axis, so a benchmark delta is
# attributable.  Baseline result: mean fwd speed +0.483 m/s vs vx=+0.80 command
# (it stays upright but undershoots — the policy is stable but lazy).  Keep PPO
# hyperparameters identical to the baseline across experiments; only the shaping
# below changes, so `play --vx 0.8` numbers are comparable.  Run e.g.:
#     python -m learning.train --config tight_tracking
#     python -m learning.play  --config tight_tracking --vx 0.8

def _tight_tracking() -> Config:
    """H1: the undershoot is a too-forgiving tracking reward.

    The exp kernel with sigma=0.2 already pays ~0.6 at 0.48 m/s, so closing the
    last 0.32 m/s buys little.  Sharpen the kernel and raise the tracking weight
    so the missing speed is genuinely costly.
    """
    cfg = Config(name="tight_tracking")
    cfg.reward.weights["track_lin_vel"] = 3.0
    cfg.reward.weights["track_ang_vel"] = 1.0
    cfg.reward.tracking_sigma = 0.12
    return cfg


def _light_reg() -> Config:
    """H2: the regularizers cap stride energy.

    A heavy height penalty plus effort/smoothness costs suppress the dynamic
    crouch-and-push of a faster gait.  Relax them so the policy is free to move
    harder, trading some smoothness for speed.
    """
    cfg = Config(name="light_reg")
    cfg.reward.weights["base_height"] = -5.0
    cfg.reward.weights["action_rate"] = -0.005
    cfg.reward.weights["torques"] = -0.00005
    return cfg


def _more_authority() -> Config:
    """H3: the residual action scale caps deviation from the slow scripted trot.

    `residual_gait` adds scale*action on top of the scripted prior; at scale=0.3
    the policy can't lengthen the stride much.  Widen the range so it can push
    the legs further per step.
    """
    cfg = Config(name="more_authority")
    cfg.action.scale = 0.45
    return cfg


EXPERIMENTS: Dict[str, Callable[[], Config]] = {
    "baseline": default_config,
    "tight_tracking": _tight_tracking,
    "light_reg": _light_reg,
    "more_authority": _more_authority,
}


def get_config(name: str = "baseline") -> Config:
    """Look up a named experiment config; raises on an unknown name."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]()


def smoke_ify(cfg: Config) -> Config:
    """
    Shrink any config to a small, fast run that verifies the *training loop*
    end-to-end (not to produce a good policy).  Drops parallelism and horizon so
    it finishes quickly even on a CPU-only Mac.  Use it once to confirm Brax PPO
    runs and eval reward moves, then run full scale on a GPU box.
    """
    cfg.ppo.num_timesteps = 2_000_000
    cfg.ppo.num_envs = 512
    cfg.ppo.episode_length = 500
    cfg.ppo.batch_size = 256
    cfg.ppo.num_minibatches = 8
    cfg.ppo.unroll_length = 10
    cfg.ppo.num_evals = 5
    return cfg


def smoke_config() -> Config:
    """Baseline shrunk for a loop test (kept for backward compatibility)."""
    return smoke_ify(default_config())

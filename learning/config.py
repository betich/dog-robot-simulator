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

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
    """The clean, simple-but-effective starting point."""
    return Config()


def smoke_config() -> Config:
    """
    A small, fast run to verify the *training loop* end-to-end (not to produce a
    good policy).  Shrinks parallelism and horizon so it finishes quickly even on
    a CPU-only Mac.  Use it once to confirm Brax PPO runs and eval reward moves,
    then switch to default_config() on a GPU box for a real run.
    """
    cfg = Config()
    cfg.ppo.num_timesteps = 2_000_000
    cfg.ppo.num_envs = 512
    cfg.ppo.episode_length = 500
    cfg.ppo.batch_size = 256
    cfg.ppo.num_minibatches = 8
    cfg.ppo.unroll_length = 10
    cfg.ppo.num_evals = 5
    return cfg

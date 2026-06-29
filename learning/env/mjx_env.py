"""
The MJX environment (logic / orchestration).

This is robot- and reward-agnostic: it owns the simulation loop and the episode
bookkeeping, and delegates every *decision* to the functional components selected
in the config:

    action  -> ctrl        via the chosen action mapping
    physics                 mjx.step, repeated n_substeps per control step
    obs                     concatenation of the chosen observation terms
    reward                  weighted sum of the chosen reward terms
    done                    OR of the chosen termination terms
    command                 the chosen sampler, resampled on a timer

It implements the Brax `Env` interface (reset / step returning a `State`) so the
Brax PPO trainer can consume it directly.  The MuJoCo qpos/qvel/ctrl layout lives
*only* in the accessor methods below, so components speak in physical quantities.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jp
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from learning import components  # noqa: F401  (populates the registries)
from learning.config import Config
from learning.env import math_utils as mu
from learning.registry import ACTIONS, COMMANDS, OBSERVATIONS, REWARDS, TERMINATIONS
from learning.robot import RobotInfo, load_robot


class MjxEnv(Env):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.robot: RobotInfo = load_robot(cfg.robot)

        self._mjx_model = self.robot.mjx_model
        self.nu = self.robot.nu

        # Control vs sim rate: take several physics substeps per policy step.
        self._sim_dt = float(self.robot.mj_model.opt.timestep)
        self.n_substeps = max(1, round(cfg.sim.ctrl_dt / self._sim_dt))
        self.dt = self.n_substeps * self._sim_dt

        # Device-resident model constants used by components.
        self.default_pose = jp.asarray(self.robot.default_pose)
        self.nominal_qpos = jp.asarray(self.robot.nominal_qpos)
        self.ctrl_range = jp.asarray(self.robot.ctrl_range)
        self.joint_range = jp.asarray(self.robot.joint_range)
        self.torso_body_id = self.robot.torso_body_id

        # Resolve the chosen components once (names -> functions).
        self._action_fn = ACTIONS.get(cfg.action.name)
        self._command_fn = COMMANDS.get(cfg.command.name)
        self._obs_fns = [OBSERVATIONS.get(n) for n in cfg.obs.terms]
        self._reward_fns = [(n, REWARDS.get(n), float(w))
                            for n, w in cfg.reward.weights.items()]
        self._term_fns = [TERMINATIONS.get(n) for n in cfg.termination.terms]

        steps = max(1, round(cfg.command.resample_time / self.dt)) \
            if cfg.command.resample_time > 0 else 1_000_000_000
        self._resample_steps = steps

        self._obs_size = int(self.reset(jax.random.PRNGKey(0)).obs.shape[-1])

    # ── MuJoCo state accessors (the only place the qpos/qvel layout lives) ────
    def base_pos(self, data):            return data.qpos[0:3]
    def base_quat(self, data):           return data.qpos[3:7]
    def joint_pos(self, data):           return data.qpos[7:7 + self.nu]
    def joint_vel(self, data):           return data.qvel[6:6 + self.nu]
    def torques(self, data):             return data.actuator_force

    def base_lin_vel_body(self, data):
        return mu.inv_rotate(self.base_quat(data), data.qvel[0:3])

    def base_ang_vel_body(self, data):
        return mu.inv_rotate(self.base_quat(data), data.qvel[3:6])

    def projected_gravity(self, data):
        return mu.projected_gravity(self.base_quat(data))

    # ── Brax Env interface ───────────────────────────────────────────────────
    @property
    def observation_size(self) -> int:   return self._obs_size

    @property
    def action_size(self) -> int:        return self.nu

    @property
    def backend(self) -> str:            return "mjx"

    def reset(self, rng: jax.Array) -> State:
        rng, q_rng, v_rng, c_rng = jax.random.split(rng, 4)

        # Jitter ONLY the joints — never the base position/quaternion, so the
        # free-joint orientation stays a valid unit quaternion at t=0.
        joint_noise = self.cfg.reset.qpos_noise * jax.random.normal(q_rng, (self.nu,))
        qpos = self.nominal_qpos.at[7:7 + self.nu].add(joint_noise)
        qvel = self.cfg.reset.qvel_noise * jax.random.normal(v_rng, (self.robot.nv,))

        data = mjx.make_data(self._mjx_model).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._mjx_model, data)

        command = self._command_fn(c_rng, self.cfg)
        info: Dict = {
            "rng": rng,
            "command": command,
            "last_action": jp.zeros(self.nu),
            "step": jp.array(0, dtype=jp.int32),
        }
        obs = self._observe(data, info)
        metrics = {f"reward/{name}": jp.array(0.0)
                   for name, _, _ in self._reward_fns}
        return State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        data0 = state.pipeline_state
        ctrl = self._action_fn(self, action, data0, state.info)

        data = jax.lax.fori_loop(
            0, self.n_substeps,
            lambda _, d: mjx.step(self._mjx_model, d.replace(ctrl=ctrl)),
            data0,
        )

        info = dict(state.info)
        info["step"] = state.info["step"] + 1

        # Resample the command on a timer (no-op if resample_time == 0).
        rng, c_rng = jax.random.split(info["rng"])
        info["rng"] = rng
        resample = (info["step"] % self._resample_steps) == 0
        new_cmd = self._command_fn(c_rng, self.cfg)
        info["command"] = jp.where(resample, new_cmd, info["command"])

        obs = self._observe(data, info)

        reward, terms = self._reward(data, action, info)
        done = self._done(data, info)

        # Guard against physics divergence: a non-finite state would otherwise
        # slip past the termination checks (nan compares false), persist across
        # steps, and poison training with NaN gradients.  Sanitize and force a
        # reset instead.
        finite = jp.isfinite(obs).all() & jp.isfinite(reward)
        obs = jp.where(finite, obs, jp.zeros_like(obs))
        reward = jp.where(finite, reward, 0.0)
        done = jp.where(finite, done, 1.0)

        info["last_action"] = action

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done,
            info=info, metrics=metrics,
        )

    # ── internals: assemble obs / reward / done from components ───────────────
    def _observe(self, data, info) -> jp.ndarray:
        obs = jp.concatenate([jp.atleast_1d(fn(self, data, info))
                              for fn in self._obs_fns])
        if self.cfg.obs.noise_scale > 0:
            rng, key = jax.random.split(info["rng"])
            info["rng"] = rng
            obs = obs + self.cfg.obs.noise_scale * jax.random.normal(key, obs.shape)
        return obs

    def _reward(self, data, action, info):
        total = jp.array(0.0)
        terms = {}
        for name, fn, weight in self._reward_fns:
            value = fn(self, data, action, info)
            terms[name] = value
            total = total + weight * value
        return total * self.dt, terms

    def _done(self, data, info) -> jp.ndarray:
        done = jp.array(False)
        for fn in self._term_fns:
            done = jp.logical_or(done, fn(self, data, info))
        return done.astype(jp.float32)


def make_env(cfg: Config) -> MjxEnv:
    """Factory used by every entry point — build the env from a config."""
    return MjxEnv(cfg)

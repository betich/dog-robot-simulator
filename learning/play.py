"""
Entry point: roll out a trained policy and watch it.

Two modes — both reuse the *training* env, so the observations the policy sees
here are identical to training (no duplicated obs code):

  live viewer (default)   an interactive window; arrow keys drive the command.
      python -m learning.play
      # macOS needs the viewer on the main thread:
      PYTHONNOUSERSITE=1 .venv-rl/bin/mjpython -m learning.play

  video (headless)        renders frames to a file with a tracking camera; runs
                          under normal `python`, no mjpython, good on a remote box.
      python -m learning.play --video walk.gif --seconds 8 --vx 0.8

Arrow keys (live mode):
    ↑ / ↓  forward / backward      ← / →  turn left / right      Space  stop
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from learning.config import Config, default_config
from learning.env.mjx_env import make_env
from learning.train import PARAMS_PATH


def build_policy(env, cfg: Config, params):
    """Reconstruct the deterministic inference fn that matches train.py."""
    normalize = (running_statistics.normalize
                 if cfg.ppo.normalize_observations
                 else (lambda x, _: x))
    networks = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=cfg.ppo.policy_hidden,
        value_hidden_layer_sizes=cfg.ppo.value_hidden,
    )
    return ppo_networks.make_inference_fn(networks)(params, deterministic=True)


def _load(params_path: str):
    cfg = default_config()
    cfg.command.resample_time = 0.0        # hold the command fixed during playback
    env = make_env(cfg)
    params = model.load_params(params_path)
    policy = jax.jit(build_policy(env, cfg, params))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    return env, policy, reset, step


def _with_command(state, cmd):
    return state.replace(info={**state.info, "command": jp.asarray(cmd)})


# ── headless video ────────────────────────────────────────────────────────────
def run_video(params_path, out_path, seconds, command):
    try:
        import imageio
    except ImportError:
        raise SystemExit("video mode needs imageio:  pip install imageio")

    env, policy, reset, step = _load(params_path)
    mj_model = env.robot.mj_model
    mj_data = mujoco.MjData(mj_model)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.robot.torso_body_id
    cam.distance, cam.azimuth, cam.elevation = 2.5, 120.0, -20.0

    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)

    frames = []
    n = int(seconds / env.dt)
    print(f"rendering {n} frames ({seconds}s) at command {command} ...")

    speeds, resets = [], 0       # mean forward speed (reset-robust) and fall count
    for _ in range(n):
        state = _with_command(state, command)
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step(state, action)
        mjx.get_data_into(mj_data, mj_model, state.pipeline_state)
        speeds.append(float(mj_data.qvel[0]))   # world-x velocity (forward)
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render())
        if float(state.done) > 0.5:
            resets += 1
            rng, key = jax.random.split(rng)
            state = reset(key)

    renderer.close()
    imageio.mimsave(out_path, frames, fps=round(1.0 / env.dt))

    mean_vx = sum(speeds) / len(speeds)
    print(f"wrote {out_path}")
    print(f"mean forward speed: {mean_vx:+.3f} m/s   (command vx={command[0]:+.2f})   "
          f"falls/resets: {resets}")


# ── interactive viewer ────────────────────────────────────────────────────────
def _key_callback(cmd):
    bindings = {265: (0, 0.8), 264: (0, -0.5), 263: (2, 0.6), 262: (2, -0.6)}

    def cb(keycode):
        if keycode == 32:
            cmd[:] = [0.0, 0.0, 0.0]
        elif keycode in bindings:
            axis, value = bindings[keycode]
            cmd[:] = [0.0, 0.0, 0.0]
            cmd[axis] = value

    return cb


def run_viewer(params_path):
    import mujoco.viewer

    env, policy, reset, step = _load(params_path)
    mj_model = env.robot.mj_model
    mj_data = mujoco.MjData(mj_model)

    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    cmd = [0.0, 0.0, 0.0]

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=_key_callback(cmd),
    ) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            state = _with_command(state, cmd)
            rng, key = jax.random.split(rng)
            action, _ = policy(state.obs, key)
            state = step(state, action)
            mjx.get_data_into(mj_data, mj_model, state.pipeline_state)
            viewer.sync()
            if float(state.done) > 0.5:
                rng, key = jax.random.split(rng)
                state = reset(key)
            remaining = env.dt - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_PATH)
    parser.add_argument("--video", metavar="PATH",
                        help="render to this file (e.g. walk.gif) instead of a window")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--vx", type=float, default=0.8, help="forward command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="lateral command (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="yaw command (rad/s)")
    args = parser.parse_args()

    if args.video:
        run_video(args.params, args.video, args.seconds, [args.vx, args.vy, args.wz])
    else:
        run_viewer(args.params)

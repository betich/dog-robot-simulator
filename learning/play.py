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

import os
import time

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from learning.config import (
    Config, EXPERIMENTS, best_checkpoint_path, checkpoint_path, get_config,
    results_dir,
)
from learning.env.mjx_env import make_env


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


def _load(config_name: str, params_path: str = None):
    # Rebuild the *exact* config used for training (action scale / obs / reward
    # all shape the env the policy expects), then load that experiment's params.
    cfg = get_config(config_name)
    cfg.command.resample_time = 0.0        # hold the command fixed during playback
    env = make_env(cfg)
    params = model.load_params(params_path or checkpoint_path(cfg.name))
    policy = jax.jit(build_policy(env, cfg, params))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    return env, policy, reset, step


def _with_command(state, cmd):
    return state.replace(info={**state.info, "command": jp.asarray(cmd)})


# ── headless video ────────────────────────────────────────────────────────────
def run_video(config_name, out_path, seconds, command, params_path=None):
    try:
        import imageio
    except ImportError:
        raise SystemExit("video mode needs imageio:  pip install imageio")

    env, policy, reset, step = _load(config_name, params_path)
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
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=round(1.0 / env.dt))

    mean_vx = sum(speeds) / len(speeds)
    summary = (f"mean forward speed: {mean_vx:+.3f} m/s   (command vx={command[0]:+.2f})   "
               f"falls/resets: {resets}")
    print(f"wrote {out_path}")
    print(summary)

    # Record the benchmark next to the rendering so experiments stay comparable.
    # Name it after the gif (rollout.gif -> metrics.txt, rollout_best.gif ->
    # metrics_best.txt) so a --best render doesn't clobber the final's metrics.
    abs_out = os.path.abspath(out_path)
    stem = os.path.splitext(os.path.basename(abs_out))[0].replace("rollout", "metrics")
    metrics_path = os.path.join(os.path.dirname(abs_out), f"{stem}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"config:    {config_name}\n")
        f.write(f"checkpoint:{os.path.basename(params_path or checkpoint_path(config_name))}\n")
        f.write(f"command:   vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f}\n")
        f.write(f"seconds:   {seconds}\n")
        f.write(f"{summary}\n")
    print(f"wrote {metrics_path}")


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


def run_viewer(config_name, params_path=None):
    import mujoco.viewer

    env, policy, reset, step = _load(config_name, params_path)
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
    parser.add_argument("--config", default="baseline", choices=sorted(EXPERIMENTS),
                        help="which experiment policy to play (see learning/config.py)")
    parser.add_argument("--best", action="store_true",
                        help="load the best-eval checkpoint (.best) instead of the final")
    parser.add_argument("--params", default=None,
                        help="override checkpoint path (defaults to the config's)")
    parser.add_argument("--video", metavar="PATH", nargs="?", const="",
                        help="render to a file instead of a window; with no path, "
                             "writes results/<config>/rollout.gif")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--vx", type=float, default=0.8, help="forward command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="lateral command (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="yaw command (rad/s)")
    args = parser.parse_args()

    params_path = args.params or (best_checkpoint_path(args.config) if args.best
                                  else checkpoint_path(args.config))

    if args.video is not None:
        suffix = "_best" if args.best else ""
        out_path = args.video or os.path.join(
            results_dir(args.config), f"rollout{suffix}.gif")
        run_video(args.config, out_path, args.seconds,
                  [args.vx, args.vy, args.wz], params_path=params_path)
    else:
        run_viewer(args.config, params_path)

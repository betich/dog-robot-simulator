"""
Entry point: roll out a trained policy and watch it.

Two modes — both reuse the *training* env, so the observations the policy sees
here are identical to training (no duplicated obs code):

  live viewer (default)   a custom GLFW window with an on-screen control panel:
                          click the buttons (or press the letter keys) to drive.
                          Runs under plain `python` — NOT mjpython, which reserves
                          the macOS main thread and would break GLFW window setup.
      PYTHONNOUSERSITE=1 .venv-rl/bin/python -m learning.play      # or: ./run.sh

  video (headless)        renders frames to a file with a tracking camera; also
                          plain `python`, good on a remote box.
      python -m learning.play --video walk.gif --seconds 8 --vx 0.8

Driving (live mode) — the full locomotion loop.  Click a button or press its key;
each sets the whole (vx, vy, wz) command, held until the next input.  Magnitudes
read from the config's command ranges, so they match what the policy was trained
on (e.g. R / the "run" button = the trained max vx):
    W  walk forward      R  run (max speed)     S  walk backward
    A  strafe left       D  strafe right        Q  turn left   E  turn right
    X  stop
Drag to orbit the camera, scroll to zoom, Esc to quit.  Because this is our own
GLFW loop (no MuJoCo simulate underneath), the keys can't collide with anything.
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
    return env, cfg, policy, reset, step


def _with_command(state, cmd):
    return state.replace(info={**state.info, "command": jp.asarray(cmd)})


def _cmd_slug(command):
    """Filesystem-friendly label from the command, so each direction's render is
    kept side by side instead of every run clobbering a single rollout.gif.
    Only the nonzero axes appear:  [1.5,0,0]->'vx+1.50',  [0,0.4,0]->'vy+0.40',
    [0.8,0.4,0]->'vx+0.80_vy+0.40',  [0,0,0]->'stop'."""
    vx, vy, wz = command
    parts = []
    if vx: parts.append(f"vx{vx:+.2f}")
    if vy: parts.append(f"vy{vy:+.2f}")
    if wz: parts.append(f"wz{wz:+.2f}")
    return "_".join(parts) if parts else "stop"


# ── headless video ────────────────────────────────────────────────────────────
def run_video(config_name, out_path, seconds, command, params_path=None):
    try:
        import imageio
    except ImportError:
        raise SystemExit("video mode needs imageio:  pip install imageio")

    env, _cfg, policy, reset, step = _load(config_name, params_path)
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

    # Track all three command axes so strafe/turn are measurable, not just forward
    # (qvel[0]=world fwd-x, qvel[1]=world lateral-y, qvel[5]=yaw rate).
    vxs, vys, wzs, resets = [], [], [], 0
    for _ in range(n):
        state = _with_command(state, command)
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step(state, action)
        mjx.get_data_into(mj_data, mj_model, state.pipeline_state)
        vxs.append(float(mj_data.qvel[0]))
        vys.append(float(mj_data.qvel[1]))
        wzs.append(float(mj_data.qvel[5]))
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render())
        if float(state.done) > 0.5:
            resets += 1
            rng, key = jax.random.split(rng)
            state = reset(key)

    renderer.close()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=round(1.0 / env.dt))

    mean = lambda xs: sum(xs) / len(xs)
    mean_vx, mean_vy, mean_wz = mean(vxs), mean(vys), mean(wzs)
    summary = (
        f"mean vx={mean_vx:+.3f} m/s  vy={mean_vy:+.3f} m/s  wz={mean_wz:+.3f} rad/s   "
        f"(command vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f})   "
        f"falls/resets: {resets}"
    )
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


# ── interactive viewer (custom GLFW window with an on-screen button panel) ──────
# We run our OWN GLFW render loop instead of mujoco.viewer.launch_passive so we can
# draw clickable control buttons on top of the scene — the passive viewer exposes
# no widget/overlay API (only user_scn for 3-D geoms).  Two consequences:
#   • must run under plain `python` (main thread), NOT mjpython (which reserves the
#     main thread for its own Cocoa loop and runs your script on a worker thread —
#     GLFW window creation there fails on macOS).  run.sh uses .venv-rl/bin/python.
#   • there's no built-in simulate handler underneath, so the letter keys are ours
#     alone and can no longer collide with MuJoCo's Space/arrows/etc.

def _command_bindings(cfg):
    """The full-loop commands — one control-panel button (and one key) each.

    Returns (key, label, [vx, vy, wz]).  Magnitudes read from the config's command
    ranges so buttons and keys match what the policy was trained on (R = trained
    max vx, S = min/backward, …).  Single source of truth for both input paths."""
    c = cfg.command
    walk_vx = min(0.8, c.lin_vel_x[1])           # nominal cruise (clamped to ≤ max)
    return [
        ("W", "fwd",      [walk_vx,        0.0,            0.0           ]),
        ("R", "run",      [c.lin_vel_x[1], 0.0,            0.0           ]),
        ("S", "back",     [c.lin_vel_x[0], 0.0,            0.0           ]),
        ("A", "strafe L", [0.0,            c.lin_vel_y[1], 0.0           ]),
        ("D", "strafe R", [0.0,            c.lin_vel_y[0], 0.0           ]),
        ("Q", "turn L",   [0.0,            0.0,            c.ang_vel_z[1]]),
        ("E", "turn R",   [0.0,            0.0,            c.ang_vel_z[0]]),
        ("X", "stop",     [0.0,            0.0,            0.0           ]),
    ]


def _layout_buttons(binds, fb_w, fb_h):
    """Place the panel as a 2×4 grid anchored bottom-left, in framebuffer pixels
    (mjr's origin is bottom-left).  Recomputed each frame so it tracks resizes; the
    same rects are used to draw and to hit-test, so clicks always line up."""
    cols, bw, bh, pad, margin = 4, 150, 46, 12, 22
    out = []
    for i, (key, label, cmd) in enumerate(binds):
        col, row = i % cols, i // cols
        left = margin + col * (bw + pad)
        bottom = margin + (1 - row) * (bh + pad)      # row 0 sits above row 1
        out.append({"rect": mujoco.MjrRect(left, bottom, bw, bh),
                    "cmd": cmd, "label": f"{key}  {label}"})
    return out


def _hit_button(buttons, px, py):
    for b in buttons:
        r = b["rect"]
        if r.left <= px <= r.left + r.width and r.bottom <= py <= r.bottom + r.height:
            return b
    return None


def run_viewer(config_name, params_path=None):
    import glfw

    env, cfg, policy, reset, step = _load(config_name, params_path)
    mj_model = env.robot.mj_model
    mj_data = mujoco.MjData(mj_model)
    binds = _command_bindings(cfg)

    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    cmd = [0.0, 0.0, 0.0]

    if not glfw.init():
        raise SystemExit("failed to initialise GLFW")
    window = glfw.create_window(1200, 800, f"ANYmal C — {config_name}", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("failed to create a GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.robot.torso_body_id
    cam.distance, cam.azimuth, cam.elevation = 3.0, 120.0, -20.0
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(mj_model, maxgeom=10000)
    context = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # Mutable UI state shared with the GLFW input callbacks below.
    ui = {"buttons": [], "drag": None, "lastx": 0.0, "lasty": 0.0}

    def _set_cmd(new):
        cmd[:] = new
        print(f"command: vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}")

    def on_mouse_button(win, button, action, mods):
        if action == glfw.RELEASE:
            ui["drag"] = None
            return
        x, y = glfw.get_cursor_pos(win)
        ui["lastx"], ui["lasty"] = x, y
        if button == glfw.MOUSE_BUTTON_LEFT:
            # screen coords (top-left origin) → framebuffer/mjr coords (bottom-left
            # origin), accounting for Retina content scale (fb ≠ window on macOS).
            fb_w, fb_h = glfw.get_framebuffer_size(win)
            w_w, w_h = glfw.get_window_size(win)
            px = x * fb_w / max(w_w, 1)
            py = fb_h - y * fb_h / max(w_h, 1)
            hit = _hit_button(ui["buttons"], px, py)
            if hit is not None:
                _set_cmd(list(hit["cmd"]))            # click acts on press, like a key
            else:
                ui["drag"] = "rotate"                 # empty space → orbit the camera
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            ui["drag"] = "move"

    def on_mouse_move(win, x, y):
        if ui["drag"] is None:
            return
        dx, dy = x - ui["lastx"], y - ui["lasty"]
        ui["lastx"], ui["lasty"] = x, y
        _, w_h = glfw.get_window_size(win)
        act = (mujoco.mjtMouse.mjMOUSE_ROTATE_V if ui["drag"] == "rotate"
               else mujoco.mjtMouse.mjMOUSE_MOVE_V)
        mujoco.mjv_moveCamera(mj_model, act, dx / max(w_h, 1), dy / max(w_h, 1),
                              scene, cam)

    def on_scroll(win, xoff, yoff):
        mujoco.mjv_moveCamera(mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0.0, -0.05 * yoff, scene, cam)

    def on_key(win, key, scancode, action, mods):
        if action == glfw.RELEASE:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
            return
        for k, _label, c in binds:
            if key == ord(k):
                _set_cmd(list(c))
                return

    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_mouse_move)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    print("driving: click a button, or press W/R/S  A/D  Q/E  X   (Esc quits, "
          "drag to orbit, scroll to zoom)")

    while not glfw.window_should_close(window):
        t0 = time.perf_counter()
        state = _with_command(state, cmd)
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step(state, action)
        mjx.get_data_into(mj_data, mj_model, state.pipeline_state)
        if float(state.done) > 0.5:
            rng, key = jax.random.split(rng)
            state = reset(key)

        fb_w, fb_h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Draw the control panel on top of the rendered scene.  The active command
        # is highlighted green so you can see what the robot is currently tracking.
        ui["buttons"] = _layout_buttons(binds, fb_w, fb_h)
        for b in ui["buttons"]:
            on = list(b["cmd"]) == list(cmd)
            r, g, bl = (0.20, 0.55, 0.32) if on else (0.14, 0.14, 0.17)
            mujoco.mjr_label(b["rect"], mujoco.mjtFont.mjFONT_NORMAL, b["label"],
                             r, g, bl, 0.9, 1.0, 1.0, 1.0, context)
        hud = f"vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                           "command", hud, context)

        glfw.swap_buffers(window)
        glfw.poll_events()
        remaining = env.dt - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)

    glfw.terminate()


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
                             "writes results/<config>/rollout_<command>.gif")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--vx", type=float, default=0.8, help="forward command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="lateral command (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="yaw command (rad/s)")
    args = parser.parse_args()

    params_path = args.params or (best_checkpoint_path(args.config) if args.best
                                  else checkpoint_path(args.config))

    if args.video is not None:
        suffix = "_best" if args.best else ""
        # Encode the command in the default filename so per-direction renders
        # (run / back / strafe / turn) don't overwrite each other.
        slug = _cmd_slug([args.vx, args.vy, args.wz])
        out_path = args.video or os.path.join(
            results_dir(args.config), f"rollout_{slug}{suffix}.gif")
        run_video(args.config, out_path, args.seconds,
                  [args.vx, args.vy, args.wz], params_path=params_path)
    else:
        run_viewer(args.config, params_path)

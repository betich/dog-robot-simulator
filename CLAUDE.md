# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A macOS playground for an ANYmal C quadruped. It has **two independent subsystems**
that share the MuJoCo model and the gait math but run in **separate Python
environments** and never run in the same process:

1. **`sim/` — scripted simulation + Open-RMF integration.** MuJoCo (macOS) runs an
   analytic IK trot gait, bridged over ZMQ to a Docker ROS 2 (Humble) container
   running an Open-RMF fleet adapter. Documented in `README.md`.
2. **`learning/` — reinforcement learning.** MuJoCo **MJX** (JAX) + Brax PPO trains
   a walking policy headless; playback uses the MuJoCo viewer. Documented in
   `learning/README.md`.

Keep them mentally separate: different deps, different run commands, different
MuJoCo backend (CPU `mujoco` vs `mujoco.mjx`).

## The two Python environments (do not mix)

- **`sim/`**: `pip install -r requirements.txt` (mujoco, pyzmq, numpy). Runs on the
  system/Mac Python with **`mjpython`** (not `python`).
- **`learning/`**: a dedicated venv with a **pinned, same-era bundle** in
  `learning/requirements.txt` (`jax==jaxlib==0.4.34`, `flax==0.8.5`, `brax==0.10.5`,
  `mujoco==3.2.3`, `numpy<2`). These pins are load-bearing: brax 0.10 calls
  `jax.device_put_replicated` (needs jax 0.4.x), while newer flax needs new jax —
  the two can't share a jax, so **never bump one in isolation**.
  ```bash
  python -m venv .venv-rl && source .venv-rl/bin/activate
  pip install -r learning/requirements.txt
  ```

## Common commands

### Scripted sim (`sim/`)
```bash
# macOS REQUIRES mjpython for the viewer (launch_passive needs the main thread)
mjpython sim/main.py --model mujoco_menagerie/anybotics_anymal_c/scene.xml
docker compose up        # start ZMQ↔ROS2 bridge + RMF adapter (start BEFORE the sim)
docker compose build
```
Arrow keys drive the viewer; ZMQ `/cmd_vel` from Docker overrides keyboard when non-zero.

### RL (`learning/`, run from repo root with the venv active)
```bash
python -m learning.check                       # validate env wiring (run FIRST; no NN)
python -m learning.train --smoke               # short loop test (~2M steps)
python -m learning.train --steps 30000000 --envs 2048   # real CPU run (~hours)
python -m learning.train                       # full default (60M/4096; wants a GPU)
python -m learning.play --video walk.gif --vx 0.8   # headless rollout + prints mean fwd speed
# live viewer on mac (the bare `mjpython` on PATH uses broken global packages):
PYTHONNOUSERSITE=1 .venv-rl/bin/mjpython -m learning.play
```

### Validation (there is no unit-test suite)
- `python -m learning.check` is the functional smoke test for the RL env.
- `python -m py_compile <files>` to catch syntax errors without the runtime.
- ROS 2 packages build with `colcon build` inside the Docker container (see `Dockerfile.ros2`).

## Architecture that spans files

### Scripted sim ↔ ROS 2 (ZMQ contract)
- macOS **binds** PUB on `:5555` (state), Docker **binds** SUB-side on `:5556` (cmd).
  See the matched docstrings in `sim/zmq_bridge.py` and
  `ros2_ws/src/anymal_bridge/anymal_bridge/bridge_node.py`. Start Docker first so
  `:5556` exists before the sim connects.
- `sim/mujoco_sim.py` is the physics loop; `sim/gait_controller.py` is the analytic
  trot (foot trajectory → planar 2-link IK → joint targets).

### RL env (`learning/`) — registry-driven, interchangeable
- **`config.py`** is the single dial board: selects components **by name** and sets
  all hyperparameters. **`registry.py`** maps name→function for the five seams.
- **`components/`** are pure functions (observations, rewards, terminations, actions,
  commands); `components/__init__.py` imports them so the `@REGISTRY.register`
  decorators run. Add a part = new function + a name in the config.
- **`env/mjx_env.py`** is the only stateful logic: it owns the MJX sim loop/episode
  and is the **only** place the raw `qpos/qvel/ctrl` layout lives (via accessor
  methods like `base_lin_vel_body`). Implements the Brax `Env` interface so Brax PPO
  consumes it directly. **`robot.py`** is the only file touching the raw `MjModel`.
- Entry points wire config → env → algorithm: `train.py`, `play.py`, `check.py`.

### The bridge between the two subsystems
- **`learning/gait_prior.py` imports gait constants from `sim/gait_controller.py`**
  (it's a JAX port of the scripted trot, used by the `residual_gait` action so the
  policy refines the scripted walk). Single source of truth: changing
  `STRIDE_PER_V`, `L_THIGH`, the sign tables, etc. in `gait_controller.py` affects
  **both** the scripted demo and the RL prior.

## Model facts that bite if ignored

- **`standing` keyframe in `anymal_c_mjx.xml` is ground truth** for this MJCF's
  joint signs. The plain `anymal_c.xml`/`scene.xml` has **no keyframe** (sim sets the
  pose explicitly); the MJX scene (`scene_mjx.xml`) does.
- **Hind legs use opposite-signed HFE/KFE** vs the front legs (the MJCF flips their
  joint axes). Any controller must respect this — a single sign convention for all
  four legs topples the robot. Encoded in `gait_controller.py` (`HFE_DIR`/`KFE_DIR`,
  front `-1` / hind `+1`) and in the keyframe.
- **Actuator/joint order is `LF, RF, LH, RH` × `HAA, HFE, KFE`** (12 total). This
  `JOINT_ORDER` is duplicated in `sim/mujoco_sim.py` and the ROS 2 `bridge_node.py`;
  keep them in sync. The RL env derives the same order from `actuator_trnid`.
- The MJX position actuators ship `kp=100`, which is too soft for the ~45 kg robot;
  `sim/mujoco_sim.py` stiffens them at load (`ACTUATOR_KP`), and the RL side relies
  on MJX physics directly.

## RL reward shaping note

Reward terms are weighted in `config.py`. Keep regularizers as **penalties** (0 at
the good state) and let velocity tracking be the dominant **positive** — a positive
bonus for a static state (e.g. a "stand tall" bonus) creates a march-in-place
optimum that only surfaces after long training, not in a smoke run. Validate
behavior with `play --video` (it prints mean forward speed + fall count); a
body-tracking camera alone can't show whether the robot is actually translating.

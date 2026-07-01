# `learning/` — RL walk-cycle environment (MuJoCo MJX + Brax PPO)

A small, deliberately-seamed reinforcement-learning setup for teaching ANYmal C
to walk. The design goal: **every part is clear by what it does, parts are
interchangeable, and one entry point wires the functional pieces into the
logic.** Start simple, swap pieces as you mature.

## The shape of it

```
                 config.py  ── the dial board (names + hyperparameters)
                     │  selects components by name
                     ▼
   registry.py  ── name → function   (the interchange mechanism)
                     ▲
   ┌─────────────────┴───────────────── FUNCTIONAL parts (pure functions) ──┐
   │ components/observations.py   state → sensor vector                      │
   │ components/rewards.py        state → reward term                        │
   │ components/terminations.py   state → done?                              │
   │ components/actions.py        action → actuator command                  │
   │ components/commands.py       rng → task command                         │
   └────────────────────────────────────────────────────────────────────────┘
                     │  looked up + composed by
                     ▼
   env/mjx_env.py  ── LOGIC: owns the sim loop + episode, delegates decisions
   robot.py        ── the only file that touches the raw MuJoCo model
                     │
                     ▼
   train.py / play.py / check.py  ── ENTRY POINTS (wire env + algorithm)
```

**Functional vs logic.** The `components/` functions are pure: `(env, data, …) →
array`. They hold *no* state and never step physics — they only read semantic
quantities (`env.base_lin_vel_body(data)`, `env.joint_pos(data)`). The *logic*
(`env/mjx_env.py`) owns the simulation, the episode, and the act of composing
components. The raw `qpos/qvel/ctrl` layout lives in exactly one place — the
accessor methods on `MjxEnv` — so a sensor or reward never indexes a raw array.

**Interchange.** To change behaviour you change a *name* or a *weight* in
`config.py`, or add a new function in `components/` and reference it. Nothing in
the env hard-codes a reward, observation, action mapping, or task.

## Setup

```bash
python -m venv .venv-rl && source .venv-rl/bin/activate
pip install -r learning/requirements.txt
```

JAX CPU wheels work on macOS for development. Real training throughput needs many
thousands of parallel envs — do long runs on a CUDA machine with `jax[cuda12]`.

## Run it (from the repo root)

```bash
python -m learning.check     # 1. validate wiring — random rollout, prints shapes
python -m learning.train     # 2. train PPO, saves learning/checkpoints/baseline/anymal_ppo
python -m learning.play      # 3. watch the policy in the MuJoCo viewer
```

Run `check` first — it builds the env and does a random rollout with no neural
net, so it isolates "does the env work" from "does training work".

**Live viewer the easy way.** `./run.sh <config>` (from the repo root) opens a
viewer with an **on-screen control panel** and selects which trained model to
drive:

```bash
./run.sh                 # default config (baseline)
./run.sh full_loop       # the full-loop policy
./run.sh full_loop --best
```

The window is a small custom GLFW viewer (not `launch_passive`) so it can draw
clickable buttons over the scene — **click a button** to drive, or press its
letter key; the active command is highlighted green and echoed top-left. Each
input sets the whole `(vx, vy, wz)` command and holds it until the next one;
magnitudes read from the config's command ranges, so they match what the policy
was trained on.

```
  W  walk forward      R  run (max speed)     S  walk backward
  A  strafe left       D  strafe right        Q  turn left   E  turn right
  X  stop
```

Drag to orbit the camera, scroll to zoom, **Esc** to quit. Because we own the
GLFW loop (no MuJoCo `simulate` underneath), the keys can't collide with anything
— and it runs under the venv's plain `python`, **not** `mjpython` (which reserves
the macOS main thread that GLFW needs). `run.sh` handles that for you.

**Experiments.** `train`/`play` take `--config <name>` to select a named config
from `config.py` (`baseline`, `tight_tracking`, `light_reg`, `more_authority`,
`full_loop`). Each changes one shaping axis to chase the velocity-tracking
benchmark; checkpoints go to `learning/checkpoints/<name>/`, renderings + metrics
to `results/<name>/`. See [`results/README.md`](../results/README.md).

**`full_loop`** is the config for the complete locomotion loop — walk, run,
backward, strafe L/R, turn L/R. The `baseline`-class configs undershoot because
uniform command sampling never trains the *extremes*; `full_loop` swaps in the
`velocity_modes` sampler (frequently draws each canonical full-magnitude command)
and widens `lin_vel_x` to `(-0.6, 1.5)` so "run" is a genuinely faster, trained
speed. Reward shaping stays at baseline (the only one that produced real
translation). Retrain it, then drive it live:

```bash
python -m learning.train --config full_loop          # retrain (wants a GPU)
./run.sh full_loop                                    # live viewer (see keys below)
```

```bash
python -m learning.train --config tight_tracking
python -m learning.play  --config tight_tracking --video --vx 0.8        # final policy
python -m learning.play  --config tight_tracking --best --video --vx 0.8  # best-eval policy
```

`train` saves the **final** params to `learning/checkpoints/<name>/anymal_ppo` and
the **best-eval** params to `…/anymal_ppo.best` (eval curves are non-monotonic, so
the final policy can be worse than the peak — `--best` plays the peak).

For a quick CPU smoke test of training, shrink the run in `config.py` (e.g.
`PPOConfig.num_envs = 256`, `num_timesteps = 2_000_000`).

## How to swap each part

| Want to change…        | Do this                                                              |
|------------------------|---------------------------------------------------------------------|
| What the policy senses | edit `config.obs.terms` (names from `components/observations.py`)    |
| The reward             | edit `config.reward.weights`, or add a term in `components/rewards.py` |
| How actions are applied| `config.action.name`: `residual_pose` / `direct_pose` / `torque`     |
| The task               | `config.command.*` ranges, or a new sampler in `components/commands.py` |
| When an episode ends    | `config.termination.*`                                              |
| The PPO hyperparameters| `config.ppo.*`                                                       |
| The robot              | write another loader returning a `RobotInfo` (see `robot.py`)       |
| The RL algorithm       | swap the trainer in `train.py` (env + components stay put)          |

## Action mappings (the policy ↔ actuator seam)

- **`residual_gait`** (default): `ctrl = scripted_trot(cmd, phase) + scale · action`.
  The policy refines a *walking* prior — the scripted trot from
  `sim/gait_controller.py`, ported to jax in `gait_prior.py` (same numbers, one
  source of truth). This is where the scripted and learned sides meet.
- **`residual_pose`**: `ctrl = standing_pose + scale · action`. No gait prior — the
  policy must discover locomotion from a static stance (tends to find a low
  crouch-shuffle without long training).
- **`direct_pose`**: same seam, kept separate so it can diverge (e.g. ignore the
  nominal pose).
- **`torque`**: joint torques — requires switching the MJCF `<actuator>` entries
  to `<motor>` first. A seam for when you move to torque-control RL.

## Status / caveats

This was scaffolded without a local JAX/MuJoCo runtime, so the dependency-free
core (config, registry, composition) is verified but the JAX/MJX/Brax paths are
**not yet executed**. Expect to pin versions (`mujoco`, `brax`, `jax` move fast)
and possibly adjust a Brax PPO kwarg. `python -m learning.check` is the fastest
way to surface any such issue.

# RL experiments

Velocity-tracking experiments for the `learning/` walking policy. Each experiment
is a named config in [`learning/config.py`](../learning/config.py) that builds on
the **baseline** and changes **one axis**, so any benchmark delta is attributable.
PPO hyperparameters are identical across experiments — only the reward/action
shaping changes — so the `play --vx 0.8` numbers below are directly comparable.

## Benchmark

`play --video --vx 0.8` rolls out the policy and prints **mean forward speed**
(world-x velocity, reset-robust) against the **+0.80 m/s** command, plus fall
count. The baseline stays upright but undershoots badly — it's stable but lazy.

| Experiment       | Hypothesis (one-line)                                        | mean fwd speed | falls |
| ---------------- | ------------------------------------------------------------ | -------------- | ----- |
| `baseline`       | starting point                                               | **+0.483 m/s** | 0     |
| `tight_tracking` | undershoot is a too-forgiving tracking reward — sharpen it   | _TBD_          | _TBD_ |
| `light_reg`      | regularizers cap stride energy — relax height/effort/rate    | _TBD_          | _TBD_ |
| `more_authority` | residual action scale caps stride — widen it (0.3 → 0.45)    | _TBD_          | _TBD_ |

Fill in a row after running each experiment (the numbers are written to
`results/<name>/metrics.txt` automatically).

## What each experiment changes vs baseline

- **`tight_tracking`** — `track_lin_vel` 2.0 → 3.0, `track_ang_vel` 0.8 → 1.0,
  `tracking_sigma` 0.2 → 0.12. The exp kernel at sigma=0.2 already pays ~0.6 at
  0.48 m/s, so closing the last 0.32 m/s buys little; a tighter kernel + higher
  weight makes the missing speed costly.
- **`light_reg`** — `base_height` -15 → -5, `action_rate` -0.01 → -0.005,
  `torques` -1e-4 → -5e-5. Heavy regularizers suppress the dynamic crouch-and-push
  of a faster gait; relaxing them trades some smoothness for speed.
- **`more_authority`** — `action.scale` 0.3 → 0.45. `residual_gait` adds
  `scale*action` on top of the scripted prior; a small scale can't lengthen the
  stride. Widen it so the policy can push the legs further per step.

## Running an experiment

From the repo root with the RL venv active:

```bash
# train (full scale wants a GPU; add --smoke or --steps/--envs for a CPU test)
python -m learning.train --config tight_tracking

# render + benchmark (writes results/tight_tracking/{rollout.gif,metrics.txt})
python -m learning.play --config tight_tracking --video --vx 0.8
```

Checkpoints land in `learning/checkpoints/<name>/anymal_ppo`; renderings and
metrics land in `results/<name>/`.

## Layout

```
results/
  README.md            # this file
  baseline/            # rollout.gif + metrics.txt for the starting policy
  <experiment>/        # created by `play --config <experiment> --video`
  archive/             # earlier ad-hoc renderings (walk*.gif)
```

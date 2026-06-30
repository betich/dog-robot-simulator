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

| Experiment              | Change vs baseline            | mean fwd speed | falls | eval_reward | verdict |
| ----------------------- | ----------------------------- | -------------- | ----- | ----------- | ------- |
| `baseline`              | —                             | **+0.483 m/s** | 0     | —           | **best** |
| `more_authority` (60M)  | `action.scale` 0.3→0.45       | +0.117 m/s     | 8     | negative    | unstable, topples |
| `light_reg` **best** @13.8M | relaxed regularizers      | +0.075 m/s     | 0     | **15.9**    | ~stationary |
| `light_reg` (60M final) | relaxed regularizers          | +0.046 m/s     | 0     | ~0.8        | stands still |
| `tight_tracking` (60M)  | `track`↑, `sigma` 0.2→0.12    | +0.042 m/s     | 0     | ~3.0        | stands still |

Numbers from `results/<name>/metrics.txt` (`play --vx 0.8`). 4096 envs.

### Verdict: the undershoot is a reward/command-design problem, not training

**None of the three shaping ideas beat baseline, and the cause is now clear — it
isn't overtraining, checkpoint selection, or any of the three axes we tried.**

The decisive test: we added best-eval checkpointing (below) and retrained
`light_reg`. Its **best** checkpoint hit `eval_reward` **15.9** with `ep_len` 834
(near-perfect survival — the highest eval anywhere) yet still walks at only
**+0.075 m/s** at the `vx=0.8` benchmark. So **`eval_reward` is decoupled from
forward speed**: the policy maximizes return over the *sampled command
distribution* (`vx ∈ [-0.5, 1.0]`, plus `vy`, `wz` — averaging near zero, including
backward/turning), which it satisfies while nearly stationary. Forcing `vx=0.8` at
playback probes a rare tail it never had to learn. That also explains why
relaxing regularizers (`light_reg`) made things *worse*, not better: with less
penalty pressure the cheapest high-return behavior is to stand and track the easy
commands.

Per-experiment notes (still true, but secondary to the above):

- **`tight_tracking`** — sharpening `sigma` to 0.12 made the exp kernel near-zero
  *and flat* for any real error → no gradient to chase 0.8 → stands. Don't tighten
  sigma; broaden it.
- **`light_reg`** — the relaxed penalties remove stability pressure; the policy
  stands rather than commit to a fast, fall-prone gait.
- **`more_authority`** — `scale=0.45` is just unstable (reward negative, `ep_len`
  ~48, 8 falls). Not an overtraining/checkpoint issue.

Caveat: `eval_reward` is **not comparable across configs** (each changed the reward
weights), and CPU runs aren't bit-reproducible (the 25M `light_reg` rerun peaked at
15.9 @ 13.8M vs the 60M run's 5.8 @ 10M). Trust the `play --vx 0.8` benchmark, not
`eval_reward`.

### Pipeline fix kept: best-eval checkpointing

Implemented and validated even though it didn't rescue the benchmark — it's still
correct hygiene (eval curves are non-monotonic, and `train.py` otherwise returns
only the final params). `train.py` now also saves the highest-eval-reward params to
`<ckpt>.best`; play with `play --config <name> --best`.

### The one direction that follows from the evidence

**Fix the command distribution, not the reward shape.** Training rarely sees
`vx=0.8`, so the policy never learns it. Options, in order:

1. Sample `vx` biased toward the high end (or a curriculum that ramps the range up),
   so the benchmark speed is actually trained — this is the direct fix.
2. Or change the benchmark to match the training distribution if 0.8 isn't the real
   target. (The mismatch itself is the bug — pick which side to move.)

Reward-weight and action-scale tuning are red herrings until the policy is actually
asked to go 0.8 during training.

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

# render + benchmark the FINAL policy (writes results/tight_tracking/{rollout.gif,metrics.txt})
python -m learning.play --config tight_tracking --video --vx 0.8
# ...or the best-eval policy (writes rollout_best.gif / metrics_best.txt)
python -m learning.play --config tight_tracking --best --video --vx 0.8
```

Checkpoints land in `learning/checkpoints/<name>/{anymal_ppo,anymal_ppo.best}`;
renderings and metrics land in `results/<name>/`.

## Layout

```
results/
  README.md            # this file
  baseline/            # rollout.gif (README demo) + metrics.txt — kept
  <experiment>/        # metrics.txt / metrics_best.txt — the numbers of record
  archive/             # earlier ad-hoc renderings (walk*.gif)
```

The large per-experiment `rollout*.gif` renders were deleted after benchmarking
(all showed a ~stationary robot); `metrics*.txt` preserves the results. Re-render
any of them with the `play` commands above.

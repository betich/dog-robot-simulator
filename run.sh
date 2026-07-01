#!/usr/bin/env bash
#
# Launch the RL walking policy in the live MuJoCo viewer (macOS).
#
# Selects which trained model version to run and opens the custom GLFW viewer with
# the on-screen control panel.  Uses the RL venv's plain `python` (NOT mjpython):
# our viewer owns its GLFW loop and needs the process main thread, whereas mjpython
# reserves the main thread for its own Cocoa loop.  PYTHONNOUSERSITE=1 keeps the
# broken ~/.local global packages out of the way (see CLAUDE.md).
#
#   ./run.sh                 # default config (baseline)
#   ./run.sh full_loop       # the full locomotion-loop policy
#   ./run.sh full_loop --best        # best-eval checkpoint
#   ./run.sh full_loop --video out.gif --vx 1.5   # passes extra args through
#
# In the viewer: click a button (or press its key) to drive —
#   W fwd  R run  S back  A/D strafe L/R  Q/E turn L/R  X stop
#   drag to orbit, scroll to zoom, Esc to quit.
#
set -euo pipefail

cd "$(dirname "$0")"

PYBIN=".venv-rl/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "error: $PYBIN not found. Create the RL venv first (see learning/README.md):" >&2
  echo "  python -m venv .venv-rl && source .venv-rl/bin/activate && pip install -r learning/requirements.txt" >&2
  exit 1
fi

# First positional arg (if it doesn't start with '-') is the config name.
CONFIG="baseline"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"; shift
fi

exec env PYTHONNOUSERSITE=1 "$PYBIN" -m learning.play --config "$CONFIG" "$@"

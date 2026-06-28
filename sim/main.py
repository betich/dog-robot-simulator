"""
Entry point for the ANYmal C MuJoCo simulation (runs on macOS).

Usage:
    python sim/main.py --model /path/to/mujoco_menagerie/anybotics_anymal_c/scene.xml

Requirements (macOS):
    pip install -r requirements.txt

Ensure the Docker ROS 2 service is running before starting so ZMQ connections
can be established immediately.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gait_controller import GaitController
from mujoco_sim import MuJoCoSim
from zmq_bridge import ZMQBridge


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get(
            "ANYMAL_MODEL",
            "mujoco_menagerie/anybotics_anymal_c/scene.xml",
        ),
        help="Path to ANYmal C scene.xml",
    )
    p.add_argument(
        "--control-freq",
        type=float,
        default=200.0,
        help="Gait controller frequency in Hz (default: 200)",
    )
    return p.parse_args()


def main():
    args  = parse_args()
    bridge = ZMQBridge()
    gait   = GaitController(control_freq=args.control_freq)
    sim    = MuJoCoSim(args.model, bridge, gait)

    try:
        sim.run()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()

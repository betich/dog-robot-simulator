# Dog Robot Simulation

This is my playground for trying MuJoCo + Open-RMF Simulation robot simulation on MacOS.

![Trained walking policy](results/baseline/rollout.gif)

```
MuJoCo (macOS)          ZMQ          Docker (ROS 2 Humble)
┌──────────────┐  state :5555 →  ┌──────────────────┐
│ sim/main.py  │                  │  bridge_node     │ → /odom, /tf, /joint_states
│ gait_ctrl    │  ← cmd  :5556   │                  │ ← /cmd_vel
└──────────────┘                  └──────────────────┘
                                  ┌──────────────────┐
                                  │  fleet_adapter   │ ← RMF path requests
                                  └──────────────────┘
```

## Prerequisites

| Tool                 | Version |
| -------------------- | ------- |
| Python               | 3.10+   |
| MuJoCo               | 3.1.0+  |
| Docker Desktop (Mac) | latest  |

## Setup

### 1. Clone this repo and get the ANYmal C model

```bash
git clone <this-repo>
cd robot-simulation
```

Download only the ANYmal C model using sparse-checkout (avoids pulling the entire ~1 GB menagerie):

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie
cd mujoco_menagerie
git sparse-checkout set anybotics_anymal_c
git checkout
cd ..
```

### 2. Install macOS Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the Docker image

```bash
docker compose build
```

## Running

> **Order matters.** Start Docker first so port 5556 is bound before the macOS sim tries to connect.

### Step 1 — Start ROS 2 services (Docker)

```bash
docker compose up
```

Starts two containers:

- `anymal_bridge` — ZMQ ↔ ROS 2 bridge, publishes `/odom`, `/tf`, `/joint_states`, forwards `/cmd_vel`
- `rmf_adapter` — Open-RMF fleet adapter, subscribes to path requests and drives `/cmd_vel`

### Step 2 — Start the MuJoCo simulation (macOS)

```bash
mjpython sim/main.py --model mujoco_menagerie/anybotics_anymal_c/scene.xml
```

> **macOS requires `mjpython`**, not `python`. `launch_passive` (the non-blocking viewer) uses a background thread for rendering, which conflicts with macOS's requirement that GUI code runs on the main thread. `mjpython` is MuJoCo's bundled launcher that works around this. It ships with the `mujoco` pip package — verify with `which mjpython`.

The MuJoCo passive viewer opens. The robot stands and runs the trot gait when commanded.

### Keyboard control (viewer window)

Click the MuJoCo viewer window to give it focus, then use:

| Key     | Action     |
| ------- | ---------- |
| `↑`     | Forward    |
| `↓`     | Backward   |
| `←`     | Turn left  |
| `→`     | Turn right |
| `Space` | Stop       |

Keyboard drives only when ZMQ/ROS 2 is idle. If Docker is sending `/cmd_vel`, that takes priority.

### Step 3 — Send a drive command via ROS 2 (optional)

In a new terminal, publish a `/cmd_vel` directly into ROS 2 inside Docker:

```bash
docker exec -it robot-simulation-anymal_bridge-1 bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.3}, angular: {z: 0.0}}" --rate 10
```

## Tuning

### Kinematic parameters (`sim/gait_controller.py`)

The IK uses approximate values. Verify these against `mujoco_menagerie/anybotics_anymal_c/*.xml` after the model is available:

| Constant       | Description                           | Default     |
| -------------- | ------------------------------------- | ----------- |
| `HIP_POS`      | Hip joint positions in body frame (m) | approximate |
| `L_THIGH`      | HFE-to-KFE link length (m)            | 0.35        |
| `L_SHANK`      | KFE-to-foot link length (m)           | 0.33        |
| `STAND_HEIGHT` | Desired body height above ground (m)  | 0.52        |
| `GAIT_FREQ`    | Trot frequency (Hz)                   | 1.5         |

### KFE sign

If knees bend the wrong direction in the viewer, flip the sign in `sim/gait_controller.py:88`:

```python
# current
q_kfe = -math.acos(cos_c)
# flip to
q_kfe = +math.acos(cos_c)
```

### Actuator names

If `mujoco_sim.py` prints a warning about missing actuators at startup, check what names the model uses:

```python
# add temporarily to MuJoCoSim.__init__
print(list(self._act_idx))
```

Then update `JOINT_ORDER` in both `sim/mujoco_sim.py` and `ros2_ws/src/anymal_bridge/anymal_bridge/bridge_node.py`.

## References

| Repo                                                                                    | Role                                 |
| --------------------------------------------------------------------------------------- | ------------------------------------ |
| [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)                     | Physics engine                       |
| [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | ANYmal C MJCF model                  |
| [open-rmf/rmf_ros2](https://github.com/open-rmf/rmf_ros2)                               | Open-RMF fleet adapter framework     |
| [open-rmf/rmf_internal_msgs](https://github.com/open-rmf/rmf_internal_msgs)             | `rmf_fleet_msgs` ROS 2 message types |

## Project structure

```
.
├── sim/
│   ├── main.py              # entry point (macOS)
│   ├── mujoco_sim.py        # MuJoCo physics loop
│   ├── gait_controller.py   # kinematic IK trot gait
│   └── zmq_bridge.py        # ZMQ pub/sub (macOS side)
├── ros2_ws/src/
│   ├── anymal_bridge/       # ZMQ ↔ ROS 2 bridge node
│   └── rmf_adapter/         # Open-RMF fleet adapter
├── config/
│   ├── fleet_config.yaml    # RMF fleet limits and footprint
│   └── nav_graph.building.yaml  # RMF building map / waypoints
├── Dockerfile.ros2
├── docker-compose.yml
└── requirements.txt
```

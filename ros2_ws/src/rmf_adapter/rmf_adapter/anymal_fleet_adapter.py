"""
Open-RMF fleet adapter for ANYmal C.

Strategy: RMF plans in 2D; this node converts the resulting path into
sequential /cmd_vel commands.  The locomotion bridge (bridge_node) translates
those into a walking gait inside MuJoCo.

Install prerequisites inside the container:
  sudo apt install -y ros-humble-rmf-fleet-msgs ros-humble-nav2-msgs

For production use, replace the simple P-controller below with:
  rmf_fleet_adapter (C++) or fleet-client (Python) from open-rmf/rmf_ros2.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rmf_fleet_msgs.msg import FleetState, RobotState, Location, PathRequest, ModeRequest


FLEET_NAME  = "anymal_fleet"
ROBOT_NAME  = "anymal_c_01"
MAX_LIN_VEL = 0.5   # m/s
MAX_ANG_VEL = 0.8   # rad/s
GOAL_TOL    = 0.15  # metres — consider waypoint reached


class AnymalFleetAdapter(Node):
    def __init__(self):
        super().__init__("anymal_fleet_adapter")

        self._cmd_pub        = self.create_publisher(Twist, "/cmd_vel", 10)
        self._fleet_state_pub = self.create_publisher(FleetState, "/fleet_states", 10)

        self._odom_sub       = self.create_subscription(
            Odometry, "/odom", self._on_odom, 10
        )
        self._path_req_sub   = self.create_subscription(
            PathRequest, "/robot_path_requests", self._on_path_request, 10
        )
        self._mode_req_sub   = self.create_subscription(
            ModeRequest, "/robot_mode_requests", self._on_mode_request, 10
        )

        self._pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._waypoints: list[Location] = []
        self._task_id  = ""

        self.create_timer(0.1, self._control_loop)       # 10 Hz nav controller
        self.create_timer(1.0, self._publish_fleet_state)
        self.get_logger().info(f"fleet adapter started — fleet={FLEET_NAME}")

    # ── odometry ──────────────────────────────────────────────────────────────
    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y**2 + q.z**2),
        )
        self._pose = {"x": p.position.x, "y": p.position.y, "yaw": yaw}

    # ── RMF path request ──────────────────────────────────────────────────────
    def _on_path_request(self, msg: PathRequest):
        if msg.fleet_name != FLEET_NAME or msg.robot_name != ROBOT_NAME:
            return
        self._waypoints = list(msg.path)
        self._task_id   = msg.task_id
        self.get_logger().info(
            f"path request  task={self._task_id}  waypoints={len(self._waypoints)}"
        )

    def _on_mode_request(self, msg: ModeRequest):
        if msg.fleet_name != FLEET_NAME or msg.robot_name != ROBOT_NAME:
            return
        if msg.mode.mode == msg.mode.MODE_PAUSED:
            self._waypoints = []
            self._stop()

    # ── 10 Hz navigation P-controller ────────────────────────────────────────
    def _control_loop(self):
        if not self._waypoints:
            return

        goal = self._waypoints[0]
        dx   = goal.x - self._pose["x"]
        dy   = goal.y - self._pose["y"]
        dist = math.hypot(dx, dy)

        if dist < GOAL_TOL:
            self._waypoints.pop(0)
            if not self._waypoints:
                self._stop()
                self.get_logger().info(f"task {self._task_id} — destination reached")
            return

        heading_err = math.atan2(dy, dx) - self._pose["yaw"]
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))  # wrap

        cmd = Twist()
        cmd.angular.z = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 2.0 * heading_err))

        # Slow down if not aligned yet
        forward_frac = max(0.0, math.cos(heading_err))
        cmd.linear.x  = MAX_LIN_VEL * forward_frac
        self._cmd_pub.publish(cmd)

    def _stop(self):
        self._cmd_pub.publish(Twist())

    # ── RMF fleet state heartbeat ─────────────────────────────────────────────
    def _publish_fleet_state(self):
        loc = Location()
        loc.x   = self._pose["x"]
        loc.y   = self._pose["y"]
        loc.yaw = self._pose["yaw"]
        loc.level_name = "L1"

        robot = RobotState()
        robot.name     = ROBOT_NAME
        robot.model    = "anymal_c"
        robot.task_id  = self._task_id
        robot.location = loc
        robot.battery_percent = 1.0

        fleet = FleetState()
        fleet.name   = FLEET_NAME
        fleet.robots = [robot]
        self._fleet_state_pub.publish(fleet)


def main(args=None):
    rclpy.init(args=args)
    node = AnymalFleetAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

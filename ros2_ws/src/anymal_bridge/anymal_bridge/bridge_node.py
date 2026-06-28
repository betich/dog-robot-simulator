"""
ZMQ ↔ ROS 2 bridge node.

Subscribes to MuJoCo state over ZMQ and republishes as standard ROS 2 topics.
Subscribes to /cmd_vel from ROS 2 and forwards to MuJoCo over ZMQ.

ZMQ addresses are configured via environment variables:
  ZMQ_STATE_ADDR  default tcp://host.docker.internal:5555  (CONNECT to macOS BIND)
  ZMQ_CMD_ADDR    default tcp://0.0.0.0:5556               (BIND here; macOS CONNECT via localhost)
"""

import json
import math
import os
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import zmq

STATE_ADDR = os.environ.get("ZMQ_STATE_ADDR", "tcp://host.docker.internal:5555")
CMD_ADDR   = os.environ.get("ZMQ_CMD_ADDR",   "tcp://0.0.0.0:5556")

ODOM_FRAME  = "odom"
BASE_FRAME  = "base_link"
JOINT_ORDER = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]


class AnymalBridgeNode(Node):
    def __init__(self):
        super().__init__("anymal_bridge")

        self._odom_pub   = self.create_publisher(Odometry,    "/odom",         10)
        self._js_pub     = self.create_publisher(JointState,  "/joint_states",  10)
        self._tf_bcast   = TransformBroadcaster(self)
        self._cmd_sub    = self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)

        ctx = zmq.Context()

        self._state_sub = ctx.socket(zmq.SUB)
        self._state_sub.connect(STATE_ADDR)
        self._state_sub.setsockopt(zmq.SUBSCRIBE, b"")

        # CMD_ADDR is owned (BIND) by this node; macOS CONNECT to the mapped port
        self._cmd_pub = ctx.socket(zmq.PUB)
        self._cmd_pub.bind(CMD_ADDR)

        self._ctx = ctx

        # Receive state in background thread; publish on ROS 2 thread via timer
        self._latest_state: dict | None = None
        self._state_lock = threading.Lock()
        threading.Thread(target=self._recv_loop, daemon=True).start()

        self.create_timer(0.02, self._publish_state)   # 50 Hz publish rate
        self.get_logger().info(f"bridge ready  state←{STATE_ADDR}  cmd→{CMD_ADDR}")

    # ── ZMQ state receiver ────────────────────────────────────────────────────
    def _recv_loop(self):
        while rclpy.ok():
            try:
                raw = self._state_sub.recv()
                state = json.loads(raw)
                with self._state_lock:
                    self._latest_state = state
            except Exception as exc:
                self.get_logger().warn(f"recv error: {exc}")

    # ── /cmd_vel → ZMQ forward ────────────────────────────────────────────────
    def _on_cmd(self, msg: Twist):
        cmd = {
            "linear_x":  msg.linear.x,
            "linear_y":  msg.linear.y,
            "angular_z": msg.angular.z,
        }
        self._cmd_pub.send(json.dumps(cmd).encode())

    # ── ZMQ state → ROS 2 topics ──────────────────────────────────────────────
    def _publish_state(self):
        with self._state_lock:
            state = self._latest_state

        if state is None:
            return

        now   = self.get_clock().now().to_msg()
        odom  = state["odom"]
        yaw   = odom["yaw"]
        half  = yaw / 2.0

        # Odometry
        om = Odometry()
        om.header.stamp    = now
        om.header.frame_id = ODOM_FRAME
        om.child_frame_id  = BASE_FRAME
        om.pose.pose.position.x    = odom["x"]
        om.pose.pose.position.y    = odom["y"]
        om.pose.pose.orientation.z = math.sin(half)
        om.pose.pose.orientation.w = math.cos(half)
        om.twist.twist.linear.x    = odom["vx"]
        om.twist.twist.linear.y    = odom["vy"]
        om.twist.twist.angular.z   = odom["wz"]
        self._odom_pub.publish(om)

        # TF odom → base_link
        tf = TransformStamped()
        tf.header.stamp            = now
        tf.header.frame_id         = ODOM_FRAME
        tf.child_frame_id          = BASE_FRAME
        tf.transform.translation.x = odom["x"]
        tf.transform.translation.y = odom["y"]
        tf.transform.rotation.z    = math.sin(half)
        tf.transform.rotation.w    = math.cos(half)
        self._tf_bcast.sendTransform(tf)

        # JointState
        js = JointState()
        js.header.stamp = now
        js.name         = JOINT_ORDER
        js.position     = [state["joint_pos"].get(n, 0.0) for n in JOINT_ORDER]
        js.velocity     = [state["joint_vel"].get(n, 0.0) for n in JOINT_ORDER]
        self._js_pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = AnymalBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
ZMQ bridge: macOS side.

Topology (each port has exactly one BIND owner):
  PUB  BIND     tcp://0.0.0.0:STATE_PORT   — macOS owns; Docker CONNECT via host.docker.internal
  SUB  CONNECT  tcp://localhost:CMD_PORT    — Docker owns (BIND inside container, port-mapped to host)
"""

import json
import os
import threading

import zmq

STATE_PORT = int(os.environ.get("ZMQ_STATE_PORT", "5555"))
CMD_HOST   = os.environ.get("ZMQ_CMD_HOST", "localhost")
CMD_PORT   = int(os.environ.get("ZMQ_CMD_PORT",   "5556"))

_EMPTY_CMD: dict = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}


class ZMQBridge:
    def __init__(self):
        self._ctx   = zmq.Context()

        self._pub   = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{STATE_PORT}")

        self._sub   = self._ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{CMD_HOST}:{CMD_PORT}")
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.setsockopt(zmq.RCVTIMEO, 0)  # non-blocking

        self._cmd_lock = threading.Lock()
        self._latest_cmd: dict = dict(_EMPTY_CMD)
        self._stop = threading.Event()

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        print(f"[zmq] state PUB BIND :{STATE_PORT}   cmd SUB CONNECT {CMD_HOST}:{CMD_PORT}")

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._sub.recv(zmq.NOBLOCK)
                cmd = json.loads(raw)
                with self._cmd_lock:
                    self._latest_cmd = cmd
            except zmq.Again:
                self._stop.wait(0.001)   # yield without busy-spinning
            except zmq.ZMQError:
                break                    # socket closed — exit cleanly

    def pub_state(self, state: dict):
        self._pub.send(json.dumps(state).encode())

    def recv_cmd(self) -> dict:
        with self._cmd_lock:
            return dict(self._latest_cmd)

    def close(self):
        self._stop.set()
        self._pub.close()
        self._sub.close()
        self._ctx.term()

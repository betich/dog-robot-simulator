#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

cd /ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
source install/setup.bash

exec "$@"

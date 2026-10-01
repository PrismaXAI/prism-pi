"""Identify which YAM arm is on which CAN bus and which wrist camera belongs to it.

For one CAN channel at a time:
  1. Connect with zero_gravity_mode=False (arm actively holds its current pose; no gripper,
     so no gripper-calibration motion).
  2. Slowly wiggle ONLY the wrist-roll joint (joint index 5) as a sine wave:
     +/- AMPLITUDE rad over PERIOD seconds, N_CYCLES cycles.
  3. Return to the exact starting pose, then close (all torques zeroed).

All other joints hold their current position throughout. Run the camera recorder
separately to capture the wiggle from all three cameras.

Usage: python identify_arms.py --channel can0
"""

import argparse
import time

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

WIGGLE_JOINT = 5  # wrist roll, last arm joint
AMPLITUDE = 0.3  # rad (~17 deg)
PERIOD = 3.0  # s per cycle
N_CYCLES = 2
RATE_HZ = 100.0

parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=str, required=True)
args = parser.parse_args()

print(f"[{args.channel}] connecting (hold current pose, no gripper)...")
robot = get_yam_robot(
    channel=args.channel,
    arm_type=ArmType.YAM,
    gripper_type=GripperType.NO_GRIPPER,
    zero_gravity_mode=False,
)
try:
    q0 = robot.get_joint_pos().copy()
    print(f"[{args.channel}] start pose (rad): {np.round(q0, 3)}")
    print(f"[{args.channel}] WIGGLE_START {time.time():.3f}")
    t_start = time.time()
    t_end = N_CYCLES * PERIOD
    while True:
        t = time.time() - t_start
        if t >= t_end:
            break
        q = q0.copy()
        q[WIGGLE_JOINT] = q0[WIGGLE_JOINT] + AMPLITUDE * np.sin(2 * np.pi * t / PERIOD)
        robot.command_joint_pos(q)
        time.sleep(1.0 / RATE_HZ)
    # settle back at the start pose before releasing
    robot.command_joint_pos(q0)
    time.sleep(0.5)
    print(f"[{args.channel}] WIGGLE_END {time.time():.3f}")
    print(f"[{args.channel}] end pose (rad): {np.round(robot.get_joint_pos(), 3)}")
finally:
    robot.close()
print(f"[{args.channel}] done, torques zeroed.")

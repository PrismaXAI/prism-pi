"""Read the raw gripper-motor angles of one arm at fully CLOSED and fully OPEN, for "grip_cal" in config/rig_mapping.json.

No motion is commanded: motor 7 (the gripper) is enabled for an instant with zero torque just to read its angle and
switched off again, exactly like the executor's start-up read. YOU move the fingers by hand between the two readings
(the gripper is limp while this runs; nothing else may be driving the arm: no executor, no teleop).

  .venv/bin/python tools/calibrate_gripper.py --side left      # LEFT arm = can1, RIGHT arm = can0 (after ./rig_up.sh)

Prints the "grip_cal" entry: [closed_raw, open_raw, train_closed, train_open]. The first two are what was just
measured; the last two are the gripper STATE values of your TRAINING DATA at closed / open (ours: left 6.351 / 1.45,
right 6.392 / 1.34, raw motor radians of the data-collection rig) and are only passed through (--train-closed /
--train-open). Run it for both sides and paste both entries.

Notes: after a motor power cycle the gripper can re-reference by a whole turn; the executor accounts for a +-2*pi
shift of both numbers at start-up, so calibrate once and keep it. The readings should differ by about 5 rad
(94 mm of finger travel); a much smaller difference means the fingers were not moved over the full stroke.
NEW in the public release and not yet exercised on hardware: check the printed numbers for plausibility.
"""
import argparse
import json

CAN = {"left": "can1", "right": "can0"}
TRAIN = {"left": (6.351, 1.45), "right": (6.392, 1.34)}


def read_gripper_raw(channel: str) -> float:
    """Brief zero-torque enable of motor 7 to read its raw angle (no motion). Same sequence as the executor's."""
    from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType
    iface = DMSingleMotorCanInterface(channel=channel, bustype="socketcan", control_mode=ControlMode.MIT)
    info = iface.motor_on(7, MotorType.DM4310)
    iface.motor_off(7)
    iface.close()
    return float(info.position)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--channel", default=None, help="CAN interface (default: can1 for left, can0 for right)")
    ap.add_argument("--train-closed", type=float, default=None, help="training-data gripper state when closed")
    ap.add_argument("--train-open", type=float, default=None, help="training-data gripper state when open")
    a = ap.parse_args()
    ch = a.channel or CAN[a.side]
    tc = a.train_closed if a.train_closed is not None else TRAIN[a.side][0]
    to = a.train_open if a.train_open is not None else TRAIN[a.side][1]
    input(f"[{a.side} / {ch}] push the fingers fully CLOSED (pads touching) and hold them there, then press Enter ")
    closed = read_gripper_raw(ch)
    print(f"   closed raw = {closed:+.3f} rad")
    input(f"[{a.side} / {ch}] pull the fingers fully OPEN and hold them there, then press Enter ")
    opened = read_gripper_raw(ch)
    print(f"   open raw   = {opened:+.3f} rad   (travel {abs(opened - closed):.2f} rad; expect about 5)")
    if abs(opened - closed) < 3.0:
        print("!! travel below 3 rad: the fingers did not move over the full stroke, or the motor wrapped; repeat")
    print("config/rig_mapping.json:")
    print(f'  "grip_cal": {{ ..., {json.dumps(a.side)}: {json.dumps([round(closed, 3), round(opened, 3), tc, to])} }}')


if __name__ == "__main__":
    main()

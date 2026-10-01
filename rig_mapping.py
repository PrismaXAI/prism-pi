"""Which physical device is what: CAN adapter serial -> interface name, camera name -> USB path, whether the two
physical arms sit on swapped sides, and the per-rig gripper calibration. Read from rig_mapping.json next to this file
(not tracked: copy rig_mapping.example.json and fill it in for your rig, see AGENTS.md). The CODE convention never
changes: LEFT arm = can1, RIGHT arm = can0, 'left' = the arm at the bottom-left of the overhead image. After any
re-cabling: run the wiggle test (tools/identify_arms.py).

Keys:
  can_by_serial         {"<USB-CAN adapter serial>": "can0" | "can1"}   (rig_up.sh renames the adapters accordingly)
  cameras               {"high" | "left_wrist" | "right_wrist": "<name under /dev/v4l/by-path/ up to the USB port>"}
  capture_fps           camera capture rate of the web UI (default 120; 60 if two cameras share one USB hub)
  swap_gripper_raw_cal  true when the physical arms are mounted on swapped sides (raw gripper calibrations swap)
  grip_cal              optional {"left": [closed_raw, open_raw, train_closed, train_open], "right": [...]} overriding
                        GRIP_CAL in dryrun_inference.py (raw motor angles of YOUR grippers, tools/calibrate_gripper.py;
                        the last two numbers are the gripper STATE values of the training data at closed / open)
"""
import json
from pathlib import Path

_DEFAULT = {"layout": "unconfigured (no rig_mapping.json)", "can_by_serial": {}, "cameras": {}, "swap_gripper_raw_cal": False}
FILE = Path(__file__).resolve().parent / "rig_mapping.json"


def load() -> dict:
    try:
        m = json.loads(FILE.read_text())
    except FileNotFoundError:
        return dict(_DEFAULT)
    return {**_DEFAULT, **m}


def camera_device(name: str) -> str:
    cams = load()["cameras"]
    if name not in cams:
        raise SystemExit(f"camera {name!r} is not mapped: copy rig_mapping.example.json to {FILE.name} and set \"cameras\" "
                         "(ls /dev/v4l/by-path/ | grep video-index0)")
    return f"/dev/v4l/by-path/{cams[name]}:1.0-video-index0"


if __name__ == "__main__":   # used by rig_up.sh:  rig_mapping.py can  -> lines "<serial> <wanted name>"
    import sys
    m = load()
    if len(sys.argv) > 1 and sys.argv[1] == "can":
        for s, n in m["can_by_serial"].items():
            print(s, n)
    else:
        print(json.dumps(m, indent=1))

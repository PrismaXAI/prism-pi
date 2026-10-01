"""Hardware access shared by the executor, the dry run and the tools.

  - PassiveYamReader: joint + gripper readout with zero torque commanded (the arm is limp)
  - Cam / HttpCam: camera frames straight from V4L2, or from the running web UI (which owns the devices)
  - CAN interface names, camera names, GRIP_CAL (raw gripper motor angle <-> training-data gripper state)
  - load_train_stats: the checkpoint's normalizer statistics (action / state ranges)

State convention (as in training): [left q1..q6, left gripper, right q1..q6, right gripper]; arm joints in joint
radians, grippers in the data rig's raw motor radians. Images: the full 16:10 sensor scaled to 640x400.
"""
import os
from pathlib import Path

import cv2
import numpy as np
from i2rt.motor_drivers.dm_driver import DMChainCanInterface

from prism_pi.rig import mapping as _rig_mapping

YAM_MOTORS = [
    [0x01, "DM4340"], [0x02, "DM4340"], [0x03, "DM4340"],
    [0x04, "DM4310"], [0x05, "DM4310"], [0x06, "DM4310"],
    [0x07, "DM4310"],  # gripper
]


class PassiveYamReader:
    """Read 6 joint positions + raw gripper motor angle with zero torque commanded.

    Motors are enabled (they answer with state) but every command is zero torque
    in MIT mode with zero gains -- the arm stays limp exactly like the SDK's
    dm_driver.py zero-command CLI.
    """

    def __init__(self, channel: str):
        self.chain = DMChainCanInterface(
            YAM_MOTORS,
            np.zeros(7),
            np.ones(7),
            channel=channel,
            motor_chain_name=f"passive_reader_{channel}",
            start_thread=False,
        )
        # same startup wrap-around correction as i2rt get_robot.py, but ARM JOINTS
        # ONLY: the gripper's calibrated raw range (-5.1..0.06) legitimately exceeds
        # -pi and must not be wrapped.
        states = self.chain.read_states()
        for idx, s in enumerate(states[:6]):
            if s.pos < -np.pi:
                self.chain.motor_offset[idx] -= 2 * np.pi
            elif s.pos > np.pi:
                self.chain.motor_offset[idx] += 2 * np.pi
        self.chain.set_commands(np.zeros(7))  # zero torque, keeps watchdog fed
        self.chain.start_thread()

    def read(self) -> np.ndarray:
        return np.array([s.pos for s in self.chain.read_states()], dtype=np.float32)

    def close(self):
        self.chain.close()


class Cam:
    def __init__(self, by_path: str, name: str):
        self.name = name
        self.cap = cv2.VideoCapture(by_path, cv2.CAP_V4L2)
        assert self.cap.isOpened(), f"cannot open {name} at {by_path}"
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):  # warm up exposure
            self.cap.read()

    def frame(self) -> np.ndarray:
        """Return HWC uint8 RGB at 640x400 (training format)."""
        # drain stale buffer, then grab fresh
        self.cap.grab()
        ok, img = self.cap.read()
        assert ok, f"read failed on {self.name}"
        assert img.shape[:2] == (1200, 1920), f"{self.name} unexpected shape {img.shape}"
        img = cv2.resize(img, (640, 400), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


UI_BASE = f"http://localhost:{os.environ.get('YAM_UI_PORT', 8080)}"   # the web UI (same YAM_UI_PORT override as prism_pi/webui.py)


class HttpCam:
    """Fetch latest frames from the running web UI (which owns /dev/video*).
    Frames are at most one capture period old (8 ms at 120 fps)."""

    def __init__(self, name: str, base: str = UI_BASE):
        self.name, self.url = name, f"{base}/frame/{name}.jpg"

    def frame(self) -> np.ndarray:
        import urllib.request
        data = urllib.request.urlopen(self.url, timeout=2).read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None, f"bad frame from {self.url}"
        if img.shape[:2] != (400, 640):
            img = cv2.resize(img, (640, 400), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def close(self):
        pass


CAM_PORTS = {"high": 2, "left_wrist": 1, "right_wrist": 3}  # camera NAMES (the keys); devices come from config/rig_mapping.json
LEFT_CAN, RIGHT_CAN = "can1", "can0"  # code convention; rig_up.sh names the adapters accordingly (config/rig_mapping.json)

# Gripper raw-motor -> training-convention mapping, calibrated 2026-09-12.
# TRAINING convention (verified in conversion code + dataset parquet):
#   state gripper: ~1.28 rad = OPEN, ~6.35 rad = CLOSED
#   action gripper: 0 = OPEN, 1 = CLOSED (i2rt normalized is the reverse: 0=closed)
# Our motors: closed at ~0, open at ~-5.1.
# Tuple = (our_closed_raw, our_open_raw, train_CLOSED, train_OPEN):
# 2026-09-13: OPEN now maps to the dataset's typical open state (L median 1.45, R 1.34), not q01
# (1.284). 1.29 falls in state-token bin 0, never seen in training, and measurably suppresses the
# model's early close decision (training-box counterfactual: close proposals 83%->62% L, 75%->46% R).
GRIP_CAL = {"left": (0.011, -5.094, 6.351, 1.45), "right": (0.060, -5.041, 6.392, 1.34)}
# The first two numbers (raw closed/open motor angles) belong to the PHYSICAL arm, the last two (dataset state values) to
# the SIDE. When the two arms are mounted on swapped sides (rig mapping), the raw calibrations swap, the side values stay.
# The values above are OUR grippers and OUR dataset: put yours in config/rig_mapping.json "grip_cal" (tools/calibrate_gripper.py).
if _rig_mapping.load().get("grip_cal"):
    GRIP_CAL = {s: tuple(float(v) for v in _rig_mapping.load()["grip_cal"][s]) for s in ("left", "right")}
    assert all(len(v) == 4 for v in GRIP_CAL.values()), "grip_cal: [closed_raw, open_raw, train_closed, train_open] per side"
if _rig_mapping.load().get("swap_gripper_raw_cal"):
    GRIP_CAL = {"left": GRIP_CAL["right"][:2] + GRIP_CAL["left"][2:], "right": GRIP_CAL["left"][:2] + GRIP_CAL["right"][2:]}


def grip_to_train(raw: float, side: str) -> float:
    oc, oo, tc, to = GRIP_CAL[side]  # closed raw -> tc (train closed), open raw -> to (train open)
    return tc + (raw - oc) * (to - tc) / (oo - oc)

STATE_NAMES = [
    "L_sh_pan", "L_sh_lift", "L_elbow", "L_wr1", "L_wr2", "L_wr3", "L_grip",
    "R_sh_pan", "R_sh_lift", "R_elbow", "R_wr1", "R_wr2", "R_wr3", "R_grip",
]


def load_train_stats(ckpt: str):
    import json, struct
    files = sorted(Path(ckpt).glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))   # step index depends on the pipeline
    assert files, f"no policy_preprocessor_step_*_normalizer_processor.safetensors in {ckpt}"
    f = open(files[0], "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = f.tell()
    out = {}
    for key, meta in hdr.items():
        if key == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        f.seek(base + s)
        dt = {"F32": np.float32, "F64": np.float64}[meta["dtype"]]
        out[key] = np.frombuffer(f.read(e - s), dtype=dt).copy()
    return out

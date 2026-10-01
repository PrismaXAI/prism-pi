"""Dry-run pi0.5 inference on the bimanual YAM rig: real cameras, real joint states,
NO actions sent to the arms (motors held at zero torque; arms stay limp).

Pipeline mirrors training exactly:
  - cameras captured at 1920x1200 MJPEG (full 16:10 sensor, same as training rigs),
    downscaled to 640x400 like the training videos
  - observation.state = [left q1..q6, left gripper RAW motor rad, right q1..q6, right gripper RAW]
    (training stats show grippers are raw motor radians, arm joints are joint radians)
  - checkpoint's own pre/post processor pipelines (rename -> normalize -> tokenize)

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python dryrun_inference.py --ckpt <checkpoint dir> \
      --task "<a task string of the checkpoint>" --rounds 3
"""

import argparse
import dataclasses
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Registers the relative-action processor step under the name our training fork saved it with.
from relative_actions_compat import RelativeActionsProcessorStep  # noqa: E402,F401  (real step under the fork's registry name)


# ---------------------------------------------------------------------------
from i2rt.motor_drivers.dm_driver import DMChainCanInterface  # noqa: E402

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


UI_BASE = f"http://localhost:{os.environ.get('YAM_UI_PORT', 8080)}"   # the web UI (same YAM_UI_PORT override as yam_webui.py)


class HttpCam:
    """Fetch latest frames from the running yam_webui.py (which owns /dev/video*).
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


ROOT = Path(__file__).resolve().parent
CAM_PORTS = {"high": 2, "left_wrist": 1, "right_wrist": 3}  # camera NAMES (the keys); devices come from rig_mapping.json
LEFT_CAN, RIGHT_CAN = "can1", "can0"  # code convention; rig_up.sh names the adapters accordingly (rig_mapping.json)

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
# the SIDE. When the two arms are mounted on swapped sides (rig_mapping.json), the raw calibrations swap, the side values stay.
# The values above are OUR grippers and OUR dataset: put yours in rig_mapping.json "grip_cal" (tools/calibrate_gripper.py).
import rig_mapping as _rig_mapping
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LeRobot pi0.5 checkpoint directory (or a shim, tools/make_ckpt_shim.py)")
    ap.add_argument("--task", required=True, help="task string, verbatim as in the training data")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "dryrun_out"))
    ap.add_argument("--cams", choices=["v4l2", "http"], default="v4l2",
                    help="http = pull frames from yam_webui.py instead of opening /dev/video*")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    print("== loading policy ==")
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    policy = PI05Policy.from_pretrained(args.ckpt)
    policy.to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, args.ckpt)
    stats = load_train_stats(args.ckpt)
    print(f"policy on cuda, {sum(p.numel() for p in policy.parameters())/1e9:.2f}B params, "
          f"VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    print(f"== opening cameras ({args.cams}) ==")
    if args.cams == "http":
        cams = {n: HttpCam(n) for n in CAM_PORTS}
    else:
        cams = {n: Cam(_rig_mapping.camera_device(n), n) for n in CAM_PORTS}

    print("== connecting arms (zero torque, passive) ==")
    left = PassiveYamReader(LEFT_CAN)
    right = PassiveYamReader(RIGHT_CAN)

    try:
        chunks, states = [], []
        for r in range(args.rounds):
            frames = {n: c.frame() for n, c in cams.items()}
            ql, qr = left.read(), right.read()
            ql[6] = grip_to_train(ql[6], "left")
            qr[6] = grip_to_train(qr[6], "right")
            state = np.concatenate([ql, qr]).astype(np.float32)
            states.append(state)

            obs = {
                "observation.images.high": frames["high"],
                "observation.images.left_wrist": frames["left_wrist"],
                "observation.images.right_wrist": frames["right_wrist"],
                "observation.state": state,
            }
            t0 = time.perf_counter()
            with torch.inference_mode():
                batch = prepare_observation_for_inference(obs, torch.device("cuda"), args.task)
                batch = pre(batch)
                chunk = policy.predict_action_chunk(batch)  # (1, 50, 14) normalized
                chunk = post(chunk)
            dt = time.perf_counter() - t0
            chunk = chunk.squeeze(0).float().cpu().numpy()  # (50, 14)
            chunks.append(chunk)
            print(f"\n-- round {r}: inference {dt*1000:.0f} ms, chunk {chunk.shape}, "
                  f"finite={np.isfinite(chunk).all()}")

            # save inputs for inspection
            for n, f in frames.items():
                cv2.imwrite(str(outdir / f"round{r}_{n}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

            # sanity report
            a_min, a_max = stats["action.min"], stats["action.max"]
            q01, q99 = stats["action.q01"], stats["action.q99"]
            first = chunk[0]
            step_delta = np.abs(np.diff(chunk, axis=0)).max(axis=0)
            print(f"{'dim':<10}{'state':>8}{'act[0]':>8}{'act_lo':>8}{'act_hi':>8}"
                  f"{'trn_q01':>9}{'trn_q99':>9}{'maxstep':>9}  flags")
            for i, nm in enumerate(STATE_NAMES):
                lo, hi = chunk[:, i].min(), chunk[:, i].max()
                flags = []
                if lo < a_min[i] - 0.05 or hi > a_max[i] + 0.05:
                    flags.append("OUT_OF_TRAIN_RANGE")
                if step_delta[i] > 0.15:
                    flags.append("JUMPY")
                print(f"{nm:<10}{state[i]:>8.3f}{first[i]:>8.3f}{lo:>8.3f}{hi:>8.3f}"
                      f"{q01[i]:>9.3f}{q99[i]:>9.3f}{step_delta[i]:>9.3f}  {','.join(flags)}")
            arm_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            print(f"first-action vs state: max |diff| over arm joints = "
                  f"{np.abs(first - state)[arm_idx].max():.3f} rad")

        np.savez(outdir / "dryrun_results.npz",
                 chunks=np.array(chunks), states=np.array(states),
                 task=args.task, ckpt=args.ckpt)
        print(f"\nsaved {args.rounds} chunks to {outdir}/dryrun_results.npz")
    finally:
        for c in cams.values():
            c.close()
        left.close()
        right.close()
        print("closed: cameras released, arm torques zeroed")


if __name__ == "__main__":
    main()

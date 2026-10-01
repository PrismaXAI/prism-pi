"""pi0.5 executor for the bimanual YAM rig, matched to the simulator's control scheme.

Default scheme (= run_policy_sim.py of the yamulator simulator):
  - A fresh observation (3 cams + 14-dim state) every --commit-ticks (25 = 0.83 s), a 50-step chunk per
    observation, chunk step i executed at observation_tick + i at 30 Hz, per-tick rate limits
    (--max-arm-step 0.10 rad, gripper 0.10), absolute clamp to the training action envelope +-0.10,
    a 15-tick ramp from the start pose, bf16 autocast, no RTC, no blending at chunk switches.
  - The sim's inference is instantaneous; on the rig it takes ~8-10 ticks. The arm is NOT held while it
    runs (that stop-and-go dwell is what --simple does, and the sim ablation showed it costs picks):
    it keeps following the previous chunk and the new chunk is entered at step (tick - observation_tick),
    so every executed step keeps the wall-clock alignment it has in the sim.
  - --simple keeps the legacy synchronous scheme (predict with the arm holding, execute steps 0..N-1,
    repeat) for A/B runs against the dwell.
  - Safety: homing to the dataset rest pose + grippers opened before the first observation, start-jump
    gate, envelope clamp, rate limits, e-stop flag (set by the web UI) checked every tick, motor-chain
    liveness, post-miss excursion abort (--abort-wr1), heartbeat JSON for the WebUI. Frames come from the
    WebUI's /frame endpoints (single owner of /dev/video*).
  - Sag compensation (sag_comp.py) is applied only to leader-action checkpoints; follower-action
    checkpoints (FOLLOWER_ACTIONS marker or "follower" in train_config.json) run without it.

Usage (arms will MOVE unless --dry):
  HF_HUB_OFFLINE=1 .venv/bin/python -m prism_pi.control.executor --ckpt <checkpoint dir> --task "<task string of the checkpoint>" --dry
"""

import argparse
import json
import queue
import signal as _signal
import threading
import time
from pathlib import Path

# E-stop kills us with SIGTERM; convert to an exception so `finally` saves the log
_signal.signal(_signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

import numpy as np
import torch

from prism_pi.paths import ESTOP_FLAG, HEARTBEAT, ROOT  # ESTOP_FLAG: latched by the web UI's e-stop; checked before and during every run
from prism_pi.policy import relative_actions_compat  # noqa: F401  registers the relative-action step name of our training fork
from prism_pi.rig.hardware import (  # noqa: F401
    CAM_PORTS,
    GRIP_CAL,
    LEFT_CAN,
    RIGHT_CAN,
    UI_BASE,
    HttpCam,
    PassiveYamReader,
    grip_to_train,
    load_train_stats,
)

CTRL_HZ = 30.0
CHUNK = 50
# per-camera training color stats (RGB mean/std over 12 random episodes)
TRAIN_COLOR = {
    "high":        (np.array([148.8, 151.3, 130.4]), np.array([93.3, 93.4, 99.0])),
    "left_wrist":  (np.array([145.3, 141.7, 113.6]), np.array([73.8, 74.5, 81.0])),
    "right_wrist": (np.array([144.0, 139.7, 126.2]), np.array([67.1, 68.3, 74.9])),
}


class SyncedFrames:
    """One /frames?fresh=1 request for all three cameras (web UI): the reply holds frames captured AFTER the
    request, so the observation is taken on fresh, mutually aligned images (<= 1 frame period old at 120 fps) and the
    joint state is read right after. Per-camera SyncedCam objects hand out the frames so the usual view wrappers apply."""

    def __init__(self, base: str = UI_BASE, timeout_ms: int = 20):
        self.url = f"{base}/frames?fresh=1&timeout_ms={timeout_ms}"
        self.latest = {}
        self.stats = {"wait_ms": [], "age_ms": []}

    def fetch(self):
        import base64, json as _json, urllib.request
        import cv2
        rep = _json.loads(urllib.request.urlopen(self.url, timeout=2).read())
        t_now = time.time()
        for n, c in rep["cams"].items():
            img = cv2.imdecode(np.frombuffer(base64.b64decode(c["jpeg"]), np.uint8), cv2.IMREAD_COLOR)
            assert img is not None, f"bad frame for {n}"
            if img.shape[:2] != (400, 640):
                img = cv2.resize(img, (640, 400), interpolation=cv2.INTER_AREA)
            self.latest[n] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.stats["age_ms"].append(1000 * (t_now - c["t"]))
        self.stats["wait_ms"].append(1000 * (rep["t_reply"] - rep["t_req"]))


class SyncedCam:
    def __init__(self, src: SyncedFrames, name: str):
        self.src, self.name = src, name

    def frame(self):
        return self.src.latest[self.name]

    def close(self):
        pass


class ViewCam:
    """Digital re-framing of one camera before the policy sees it: zoom z (>1 = crop the centre 1/z and scale back
    to 640x400) and a content shift (dx, dy) in output pixels (+dx = content moves right, +dy = content moves down).
    One affine warp; uncovered borders are edge-replicated. Used to test (a) a tighter overhead view and (b) small
    wrist-view corrections toward the dataset's finger silhouette (rig 2026-09-18: left wrist fingers ~9 px left /
    ~15 px high of the dataset's, right wrist ~7 px right / ~15 px high)."""

    def __init__(self, inner, zoom: float = 1.0, dx: float = 0.0, dy: float = 0.0):
        self.inner, self.zoom, self.dx, self.dy = inner, float(zoom), float(dx), float(dy)

    def frame(self):
        import cv2 as _cv2
        f = self.inner.frame(); h, w = f.shape[:2]; z = self.zoom
        M = np.array([[z, 0, (1 - z) * w / 2 + self.dx], [0, z, (1 - z) * h / 2 + self.dy]], np.float32)
        return _cv2.warpAffine(f, M, (w, h), flags=_cv2.INTER_LINEAR if z >= 1 else _cv2.INTER_AREA, borderMode=_cv2.BORDER_REPLICATE)

    def close(self):
        self.inner.close()


class VirtualPitchCam:
    """Software re-aim of a camera: a rotation about the optical centre is exactly a homography of the image,
    H = K R K^-1 (sim check 2026-09-19: matches a truly pitched render to ~2 grey levels). Same sign as the sim's
    --wrist-pitch / --wrist-virtual-pitch: + = the wrist camera looks further AHEAD, the fingers move DOWN in the image.
    Sim, 90 paired scenes: +6..+9 deg removes about half of the ~13 mm forward overshoot at grasp and adds 12-15 places;
    beyond +9 the filler rows at the top start to hurt. K from the lens' diagonal FOV (our 2.2 mm lenses: 114 deg ->
    f = 245 px at 640x400), principal point assumed at the image centre. Rows from outside the frame are edge-replicated."""

    def __init__(self, inner, pitch_deg: float, diag_fov_deg: float = 114.0):
        self.inner, self.pitch = inner, float(pitch_deg)
        w, h = 640, 400
        f = np.hypot(w / 2, h / 2) / np.tan(np.radians(diag_fov_deg) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        a = -np.radians(self.pitch)
        R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        self.H = K @ R @ np.linalg.inv(K)

    def frame(self):
        import cv2 as _cv2
        f = self.inner.frame()
        return _cv2.warpPerspective(np.ascontiguousarray(f), self.H, (f.shape[1], f.shape[0]), flags=_cv2.INTER_LINEAR, borderMode=_cv2.BORDER_REPLICATE)

    def close(self):
        self.inner.close()


class DegradedCam:
    """Approximate the training pipeline's 960x600->h264-crf20 softness:
    mild gaussian blur + jpeg q60 round-trip."""

    def __init__(self, inner):
        self.inner = inner

    def frame(self):
        import cv2 as _cv2
        f = _cv2.GaussianBlur(self.inner.frame(), (3, 3), 0.6)
        ok, enc = _cv2.imencode(".jpg", f, [_cv2.IMWRITE_JPEG_QUALITY, 60])
        return _cv2.imdecode(enc, _cv2.IMREAD_COLOR)  # stays RGB order end-to-end

    def close(self):
        self.inner.close()


class ColorMatchedCam:
    """Wrap HttpCam with a fixed per-channel affine map toward training color stats.
    Live stats are measured once at startup (N frames) so the mapping is constant."""

    def __init__(self, inner, name: str, n_calib: int = 10):
        self.inner, self.name = inner, name
        t_mean, t_std = TRAIN_COLOR[name]
        frames = np.stack([inner.frame().astype(np.float32) for _ in range(n_calib)])
        l_mean = frames.reshape(-1, 3).mean(0)
        l_std = frames.reshape(-1, 3).std(0) + 1e-6
        self.gain = t_std / l_std
        self.bias = t_mean - l_mean * self.gain
        print(f"color-match {name}: gain {np.round(self.gain,2)}, bias {np.round(self.bias,1)}")

    def frame(self):
        f = self.inner.frame().astype(np.float32) * self.gain + self.bias
        return np.clip(f, 0, 255).astype(np.uint8)

    def close(self):
        self.inner.close()
MAX_ARM_STEP = 0.07   # rad per control tick, per joint. 2026-09-19: lowered from the sim's 0.10 after the table crash — the demonstrations
                      # never exceed 0.082 rad/tick (p99.9 <= 0.072) and good rig runs sit at p99 ~0.03, so 0.07 only bites on abnormal jumps
MAX_GRIP_STEP = 0.10  # normalized per tick
RAMP_TICKS = 15       # 0.5 s ramp-in


def read_gripper_raw(channel: str) -> float:
    """Brief zero-torque enable of motor 7 to read its raw angle (no motion)."""
    from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType
    iface = DMSingleMotorCanInterface(channel=channel, bustype="socketcan", control_mode=ControlMode.MIT)
    info = iface.motor_on(7, MotorType.DM4310)
    iface.motor_off(7)
    iface.close()
    return info.position


class RealArm:
    """i2rt robot with calibrated gripper limits; commands [q1..q6, grip_norm]."""

    def __init__(self, channel: str, side: str):
        import numpy as np
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        self.side = side
        oc, oo, self.tc, self.to = GRIP_CAL[side]
        # get_yam_robot wraps any motor reading outside +-pi by 2*pi at startup,
        # gripper included. Shift the limits we pass by the same wrap so the
        # normalized mapping stays exact regardless of where the gripper starts.
        # After a motor power cycle the gripper motor can also re-reference by a whole turn (2026-09-18: fully open
        # read +1.25 rad instead of -5.03). So pick the 2*pi shift that puts the reading i2rt will see (raw wrapped
        # into +-pi) inside the calibrated closed..open range, and refuse to start if none (or more than one) fits.
        raw = read_gripper_raw(channel)
        r_eff = (raw + np.pi) % (2 * np.pi) - np.pi
        lo, hi = min(oc, oo) - 0.5, max(oc, oo) + 0.5
        fits = [k * 2 * np.pi for k in (-1, 0, 1) if lo + k * 2 * np.pi <= r_eff <= hi + k * 2 * np.pi]
        if len(fits) != 1:
            raise SystemExit(f"ABORT: {side} gripper raw angle {raw:+.3f} rad (effective {r_eff:+.3f}) fits {len(fits)} calibrated ranges "
                             f"[{lo:+.2f}, {hi:+.2f}] + k*2pi — recalibrate the gripper (GRIP_CAL) before running")
        shift = fits[0]
        print(f"   {side} gripper: raw {raw:+.3f} rad, limit shift {shift / (2 * np.pi):+.0f} turn(s)")
        self.robot = get_yam_robot(
            channel=channel,
            arm_type=ArmType.YAM,
            gripper_type=GripperType.LINEAR_4310,
            gripper_limits_override=np.array([oc + shift, oo + shift]),
            zero_gravity_mode=False,  # hold pose until first command
            enable_auto_recovery=True,  # clean+re-enable errored motors instead of dying
        )

    @property
    def alive(self) -> bool:
        return bool(self.robot.motor_chain.running)

    def state_train(self) -> np.ndarray:
        """[q1..q6 rad, gripper in TRAINING raw convention]."""
        q = self.robot.get_joint_pos().copy()  # gripper slot is normalized 0..1
        q[6] = self.tc + q[6] * (self.to - self.tc)
        return q

    def command(self, a7: np.ndarray):
        self.robot.command_joint_pos(a7)

    def close(self):
        self.robot.close()


class DryArm:
    """Zero-torque passive reader; commands are recorded, never sent."""

    def __init__(self, channel: str, side: str):
        self.side = side
        self.reader = PassiveYamReader(channel)
        self.sent = []

    def state_train(self) -> np.ndarray:
        q = self.reader.read().copy()
        q[6] = grip_to_train(q[6], self.side)
        return q

    @property
    def alive(self) -> bool:
        return True

    def command(self, a7: np.ndarray):
        self.sent.append(a7.copy())

    def close(self):
        self.reader.close()


class Pi05Runner:
    def __init__(self, ckpt: str, exec_horizon: int, use_rtc: bool, dtype: str):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        self.policy = PI05Policy.from_pretrained(ckpt)
        if use_rtc:
            self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=exec_horizon)
            self.policy.init_rtc_processor()
        self.exec_horizon = exec_horizon
        self.use_rtc = use_rtc
        self.policy.to("cuda").eval()
        self.autocast = dtype == "bf16"  # autocast, not model cast: matches mixed-precision training
        self.pre, self.post = make_pre_post_processors(self.policy.config, ckpt)
        from lerobot.policies.utils import prepare_observation_for_inference
        self._prep = prepare_observation_for_inference

    @torch.no_grad()  # not inference_mode: RTC guidance re-enables autograd internally
    def infer(self, frames: dict, state14: np.ndarray, task: str,
              leftover_norm: torch.Tensor | None, delay_steps: int):
        obs = {
            "observation.images.high": frames["high"],
            "observation.images.left_wrist": frames["left_wrist"],
            "observation.images.right_wrist": frames["right_wrist"],
            "observation.state": state14.astype(np.float32),
        }
        batch = self._prep(obs, torch.device("cuda"), task)
        batch = self.pre(batch)
        kw = {}
        if self.use_rtc and leftover_norm is not None:
            kw = dict(inference_delay=int(delay_steps),
                      prev_chunk_left_over=leftover_norm.to("cuda"),
                      execution_horizon=self.exec_horizon)
        from contextlib import nullcontext
        with torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else nullcontext():
            norm_chunk = self.policy.predict_action_chunk(batch, **kw)  # (1,50,14) normalized
        exec_chunk = self.post(norm_chunk.float()).squeeze(0).float().cpu().numpy()
        return norm_chunk.squeeze(0).float().cpu(), exec_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True, help="LeRobot pi0.5 checkpoint directory (or a shim, tools/make_ckpt_shim.py)")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to run (0 = until told to end via --control-file)")
    ap.add_argument("--command-substeps", type=int, default=1,
                    help="send N linearly interpolated position commands per 30 Hz tick (N=4 -> 120 Hz) instead of one step; removes "
                         "the 30 Hz step excitation of the arms (left wrist camera oscillated at 30 Hz, ~9 px) at the cost of one tick "
                         "of command lag; 1 = off (sim-matched behaviour)")
    ap.add_argument("--sync-obs-to-frame", action="store_true",
                    help="take each observation on FRESH camera frames: one /frames?fresh=1 request to the webui that waits (<= --sync-timeout-ms) "
                         "for every camera to deliver a frame captured after the request, then read the joint state; otherwise the newest "
                         "frame at the tick is used (up to one frame period stale, cameras mutually unaligned)")
    ap.add_argument("--sync-timeout-ms", type=int, default=20)
    ap.add_argument("--control-file", default=None,
                    help="JSON file polled during the run (written by the web UI): {\"cmd\": \"run\"|\"pause\"|\"end\", \"task\": str}. "
                         "pause = hold both arms in place under power, no inference (time paused does not count toward --duration); "
                         "run = resume from a fresh observation (optionally with a new task); end = park and release. "
                         "Sim-matched scheme with the policy server only.")
    ap.add_argument("--dry", action="store_true", help="read state but send NO commands")
    ap.add_argument("--rtc", action="store_true",
                    help="real-time chunking: denoise each new chunk with guidance toward the unexecuted tail of the running "
                         "plan (frozen for the steps before the expected install offset, fading out at --rtc-horizon). Runs "
                         "through the policy server, which re-references the tail to the new state for relative-action ckpts")
    ap.add_argument("--rtc-horizon", type=int, default=20, help="chunk step at which RTC guidance has faded to zero")
    ap.add_argument("--no-rtc", action="store_true", help=argparse.SUPPRESS)  # legacy no-op (RTC is off by default)
    ap.add_argument("--exec-horizon", type=int, default=16, help="RTC execution horizon (only with --rtc)")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16", help="bf16 = autocast, as in the sim")
    ap.add_argument("--max-start-jump", type=float, default=0.25,
                    help="abort if first action deviates more than this (rad) from current pose")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--run-name", default=None, help="name of the run directory under --out (default: a timestamp); the WebUI "
                    "passes its own stamp so it can find run.npz for the LeRobot dataset conversion (prism_pi/rollout_to_lerobot.py)")
    ap.add_argument("--max-arm-step", type=float, default=MAX_ARM_STEP,
                    help="per-tick arm joint rate limit (rad); default 0.07 (sim runner default is still 0.10: pass --max-arm-step 0.10 for strict sim parity)")
    ap.add_argument("--color-match", action="store_true",
                    help="affine-map live frames toward training color statistics")
    ap.add_argument("--simple", action="store_true",
                    help="LEGACY synchronous scheme: predict while the arm holds (~0.3 s dwell), execute steps 0..N-1, "
                         "repeat. Default is the sim-matched continuous scheme; use this only for A/B runs")
    ap.add_argument("--degrade", action="store_true",
                    help="soften frames to match training video quality (blur + jpeg round-trip)")
    ap.add_argument("--sag-comp", choices=["auto", "on", "off"], default="auto",
                    help="dataset follower-sag offset on arm targets (sag_comp.py, needs config/sag_table.json). auto = ON for "
                         "leader-action checkpoints when a sag table is installed, OFF if the checkpoint dir contains a "
                         "FOLLOWER_ACTIONS marker file or its train_config.json mentions follower actions (never double-compensate)")
    ap.add_argument("--abort-wr1", type=float, default=0.35,
                    help="end the run (park under torque) if either wrist_1 exceeds this many rad for >0.5 s — "
                         "the post-miss out-of-distribution signature (wrist up, shoulder high). 0 disables")
    ap.add_argument("--no-home", action="store_true",
                    help="do NOT drive the arms to the dataset rest pose [0,0,0,-0.08,0,0] before starting")
    ap.add_argument("--keep-grippers", action="store_true",
                    help="do NOT open the grippers before the first observation (default opens them)")
    ap.add_argument("--joint-offset-vec-deg", type=float, nargs=6, default=None,
                    help="encoder-offset hypothesis (sim --joint-offset-vec-deg), deg per joint [pan lift elbow wr1 wr2 wr3], both arms: "
                         "the policy sees q+delta and its commands are sent as cmd-delta. Sim pilot 2026-09-17: lift -4 or elbow +4 "
                         "removed air-closes (fingers land ~3 cm lower). Not supported with --simple")
    ap.add_argument("--arbitrate", action="store_true",
                    help="arm arbitration (arm_arbiter.py): at most one arm in the shared work zone over the plate; the other is "
                         "held at the rest pose until the owner leaves (or dithers past --arb-timeout). Reduces crowding, "
                         "wrist-camera occlusion and gripper/camera collision risk. Sim-matched scheme only")
    ap.add_argument("--arb-hold", choices=["freeze", "rest"], default="rest", help="held arm: 'rest' = return to the rest pose (zero crowding; sim: same success, first grasp +6 s); 'freeze' = stop at the zone boundary (first grasp +3 s but crowding only halved)")
    for _n in ("high", "left-wrist", "right-wrist"):
        ap.add_argument(f"--{_n}-view", type=float, nargs=3, default=None, metavar=("ZOOM", "DX", "DY"),
                        help=f"digital re-framing of the {_n} camera before the policy: zoom (>1 crops), content shift px (+x right, +y down)")
    for _n in ("left-wrist", "right-wrist"):
        ap.add_argument(f"--{_n}-virtual-pitch", type=float, default=0.0,
                        help=f"software re-aim of the {_n} camera by an image homography, deg (+ = looks further ahead, fingers lower in the image). Sim: +6..+9 is the useful range")
    ap.add_argument("--max-down-speed-cm-s", type=float, nargs=2, default=[35.0, 25.0], metavar=("HIGH", "NEAR_TABLE"),
                    help="cap on the commanded DOWNWARD fingertip speed, cm/s: HIGH applies above 15 cm, tapering to NEAR_TABLE at/below 5 cm above the arm base plane "
                         "(demonstrations: max 35, p99.9 22-31 near the table; the 2026-09-19 crash plunged at ~55). 0 0 disables. Sim-matched scheme")
    ap.add_argument("--preclose-advance-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="one-shot tool-frame advance at close onset (tool_offset.py, as in the simulator's run_policy_sim.py): when the policy starts to close a "
                         "gripper it is held for --preclose-ticks while the tool advances by this vector (mm, gripper frame: x across the pads, + = away from the "
                         "wrist camera = forward along the reach; z along the fingers, - = toward the tips), then the close proceeds. Sim-matched scheme only")
    ap.add_argument("--preclose-ticks", type=int, default=6, help="ticks (30 Hz) over which the pre-close advance is made and the gripper is held")
    ap.add_argument("--preclose-pad-floor-cm", type=float, default=3.0,
                    help="the advance may not lower the pad centre below this height above the arm base plane (the downward part is dropped; the forward part is kept)")
    ap.add_argument("--grip-assist", type=float, default=0.0,
                    help="extra closure (normalized 0..1) added to the SENT gripper command while the policy's gripper "
                         "command is closing (ramps in over cmd 0.10..0.20; large fruit is held at cmd ~0.2). The policy-frame command, rate limiter, RTC "
                         "tail and logs are untouched; only the motor target squeezes harder. 0 disables; try 0.05")
    ap.add_argument("--arb-prefer", choices=["left", "right"], default=None, help="with --arbitrate: this arm gets the work zone whenever it is free and either arm wants in (the first ownership decision is sticky: a held arm at rest stops proposing motion)")
    ap.add_argument("--arb-zone", type=float, default=0.20, help="work-zone half width, m (pad lateral distance from the rig centre)")
    ap.add_argument("--arb-timeout", type=float, default=8.0, help="s an owner may sit in the zone without closing before yielding")
    ap.add_argument("--server", choices=["auto", "on", "off"], default="auto",
                    help="use the persistent policy server (prism_pi.policy.server) so the model stays in VRAM between runs. "
                         "auto = use it if it is up, else load the model locally; on = require it; off = always load locally")
    ap.add_argument("--plan-samples", type=int, default=1,
                    help="K flow-matching samples per observation in one batched forward pass (needs the server). "
                         "Latency: K=1 ~0.20 s, 2 ~0.27 s, 4 ~0.41 s, 8 ~0.76 s")
    ap.add_argument("--plan-select", choices=["first", "consistent"], default="consistent",
                    help="with --plan-samples > 1: 'consistent' installs the sample closest to the unexecuted part of the "
                         "current plan (arm joints, overlapping wall-clock steps); 'first' ignores the extra samples")
    ap.add_argument("--commit-ticks", type=int, default=25,
                    help="ticks between observations (sim default 25); with --simple: steps executed per chunk")
    args = ap.parse_args()
    import sys as _sys0
    _sys_argv = list(_sys0.argv)

    if ESTOP_FLAG.exists():
        raise SystemExit("E-STOP flag is latched — clear it in the WebUI first.")
    if any(v != 0 for v in args.preclose_advance_mm) and args.simple:
        raise SystemExit("--preclose-advance-mm is only implemented for the sim-matched scheme (drop --simple)")
    OFF14 = np.zeros(14)
    if args.joint_offset_vec_deg is not None:
        if args.simple:
            raise SystemExit("--joint-offset-vec-deg is not supported with --simple")
        d6 = np.radians(np.array(args.joint_offset_vec_deg, float))
        OFF14[0:6] = d6; OFF14[7:13] = d6
        print(f"== encoder-offset hypothesis active: policy sees q+delta, sends cmd-delta; delta deg = {args.joint_offset_vec_deg} ==")

    # --- decide sag compensation; a follower-action checkpoint already predicts achievable positions ---
    ck = Path(args.ckpt)
    follower_ckpt = (ck / "FOLLOWER_ACTIONS").exists()
    try:
        follower_ckpt |= "follower" in (ck / "train_config.json").read_text().lower()
    except Exception:
        pass
    from prism_pi.control import sag_comp
    if args.sag_comp == "auto":
        args.no_sag_comp = follower_ckpt or not sag_comp.available()
    else:
        args.no_sag_comp = args.sag_comp == "off"
        if args.sag_comp == "on" and follower_ckpt:
            raise SystemExit("REFUSING: --sag-comp on with a follower-action checkpoint would double-compensate.")
        if args.sag_comp == "on" and not sag_comp.available():
            raise SystemExit(f"--sag-comp on needs {sag_comp.TABLE} (see config/sag_table.example.json)")
    print("== sag compensation: " + ("ON (leader-action checkpoint)" if not args.no_sag_comp else
                                     "OFF (follower-action checkpoint)" if follower_ckpt else "OFF (no config/sag_table.json)") + " ==")

    run_dir = Path(args.out) / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Prompt-tokenizer guard (2026-09-17 root cause): a substitute paligemma tokenizer seeded into the HF cache
    # produced no <bos> token, so every prompt was off-distribution and the policy hedged. Refuse to run unless
    # the tokenization matches the training rig's ([<bos>, words...]).
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    _ids = _tok(args.task, add_special_tokens=True)["input_ids"]
    print(f"== prompt tokens: {_ids} ==")
    if not _ids or _ids[0] != _tok.bos_token_id or _tok.bos_token_id != 2:
        raise SystemExit("ABORT: prompt tokenization lacks the <bos> token (id 2) — the HF cache holds a substitute "
                         "tokenizer. Install the real google/paligemma-3b-pt-224 tokenizer snapshot (see AGENTS.md).")
    use_server = False
    if args.server != "off":
        from prism_pi.policy.client import RemoteRunner, server_info
        if server_info() is not None:
            use_server = True
        elif args.server == "on":
            raise SystemExit("ABORT: --server on but no policy server is reachable (start python -m prism_pi.policy.server)")
    if args.plan_samples > 1 and not use_server:
        raise SystemExit("ABORT: --plan-samples > 1 needs the policy server")
    if args.rtc and use_server:
        print(f"== RTC via the policy server: guidance horizon {args.rtc_horizon} steps ==")
    if use_server:
        runner = RemoteRunner(args.ckpt, args.dtype)
        print(f"== policy server: {runner.info['ckpt']} ({runner.info['dtype']}), relative={runner.info['relative']}, "
              f"{runner.info.get('n_infer', 0)} inferences served so far ==")
    else:
        print("== loading policy locally (no policy server) ==")
        runner = Pi05Runner(args.ckpt, args.exec_horizon, args.rtc, args.dtype)
    stats = load_train_stats(args.ckpt)
    # Absolute target envelope. A relative-action checkpoint stores DELTA ranges as its action stats, which must not
    # be used as an absolute clamp: take the follower STATE range instead (same as the follower-action range on the
    # arm joints). Grippers are overridden to 0..1 just below.
    is_relative = bool(json.loads((Path(args.ckpt) / "config.json").read_text()).get("use_relative_actions", False))
    env_key = "observation.state" if is_relative else "action"
    a_lo = stats[f"{env_key}.min"][:14] - 0.10
    a_hi = stats[f"{env_key}.max"][:14] + 0.10
    print(f"== action envelope from {env_key} stats (relative-action checkpoint: {is_relative}) ==")
    a_lo[6] = a_lo[13] = 0.0
    a_hi[6] = a_hi[13] = 1.0

    synced = None
    if args.sync_obs_to_frame:
        synced = SyncedFrames(timeout_ms=args.sync_timeout_ms)
        synced.fetch()
        print(f"== cameras (via webui, FRAME-SYNCED observations: fresh frames within {args.sync_timeout_ms} ms, first wait {synced.stats['wait_ms'][-1]:.1f} ms) ==")
        cams = {n: SyncedCam(synced, n) for n in CAM_PORTS}
    else:
        print("== cameras (via webui) ==")
        cams = {n: HttpCam(n) for n in CAM_PORTS}
    for n in list(cams):
        v = getattr(args, f"{n}_view", None)
        if v is not None:
            cams[n] = ViewCam(cams[n], *v)
            print(f"== {n}: digital view zoom {v[0]:.3f}, content shift ({v[1]:+.0f}, {v[2]:+.0f}) px ==")
        vp = float(getattr(args, f"{n}_virtual_pitch", 0.0) or 0.0)
        if vp != 0.0:
            cams[n] = VirtualPitchCam(cams[n], vp)
            print(f"== {n}: virtual pitch {vp:+.1f} deg (homography re-aim) ==")
    for c in cams.values():
        c.frame()  # fail fast if webui down
    if args.color_match:
        cams = {n: ColorMatchedCam(c, n) for n, c in cams.items()}
    if args.degrade:
        cams = {n: DegradedCam(c) for n, c in cams.items()}

    print(f"== arms ({'DRY - no commands' if args.dry else 'LIVE'}) ==")
    Arm = DryArm if args.dry else RealArm
    left, right = Arm(LEFT_CAN, "left"), Arm(RIGHT_CAN, "right")

    log = {"chunks": [], "offsets": [], "latency": [], "cmds": [], "states": [], "infer_tick": [], "install_tick": [], "plan_sel": [], "plan_costs": [], "arb_owner": [], "grip_sent": [], "tool_shift_mm": [], "sent_arm": [],
           # per-tick wall clock of the state read + segment id (sim-matched scheme only): rollout_to_lerobot.py pairs every tick with
           # the WebUI's frame recording by these times and cuts one LeRobot episode per segment (a PAUSE/RESUME = scene reset = new segment)
           "t_wall": [], "segment": []}
    segments = []   # [{"segment", "task", "tick", "t_wall"}], one entry per run segment (segment 0 = the start, +1 at every RESUME)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_ctr = [0]

    def save_frames(frames: dict):
        """JPEG-dump the exact frames given to the policy (RGB HWC 640x400) for post-mortem."""
        import cv2 as _cv2
        k = frame_ctr[0]; frame_ctr[0] += 1
        for n, f in frames.items():
            _cv2.imwrite(str(frames_dir / f"{k:04d}_{n}.jpg"), _cv2.cvtColor(f, _cv2.COLOR_RGB2BGR),
                         [_cv2.IMWRITE_JPEG_QUALITY, 85])
    result_q: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()

    def get_state():
        return np.concatenate([left.state_train(), right.state_train()])

    REST_POSE = np.array([0, 0, 0, -0.08, 0, 0], dtype=float)  # dataset episode start, both arms
    if not args.no_home and not args.dry:
        # Every training episode starts from this exact pose. Manual repositioning left wr1 at
        # -0.60 in run 7 (wrist camera pitched 30 deg steeper than the data at the first frame).
        q = get_state()
        dev = np.abs(np.r_[q[0:6] - REST_POSE, q[7:13] - REST_POSE])
        print(f"== homing to dataset rest pose (max deviation {dev.max():.2f} rad) ==")
        if dev.max() > 0.7:
            raise SystemExit("ABORT: an arm joint is >0.7 rad from the rest pose — move the arms near "
                             "[0,0,0,-0.08,0,0] by hand first.")
        gl = np.clip((q[6] - GRIP_CAL["left"][3]) / (GRIP_CAL["left"][2] - GRIP_CAL["left"][3]), 0, 1)
        gr = np.clip((q[13] - GRIP_CAL["right"][3]) / (GRIP_CAL["right"][2] - GRIP_CAL["right"][3]), 0, 1)
        cl, cr = q[0:6].copy(), q[7:13].copy()
        HOME_STEP = 0.03  # rad per tick, gentle
        while max(np.abs(cl - REST_POSE).max(), np.abs(cr - REST_POSE).max()) > 0.005:
            if ESTOP_FLAG.exists():
                raise SystemExit("E-STOP during homing")
            cl = cl + np.clip(REST_POSE - cl, -HOME_STEP, HOME_STEP)
            cr = cr + np.clip(REST_POSE - cr, -HOME_STEP, HOME_STEP)
            left.command(np.concatenate([cl, [1.0 - gl]]))
            right.command(np.concatenate([cr, [1.0 - gr]]))
            time.sleep(1.0 / CTRL_HZ)
        time.sleep(0.3)
        print("   at rest pose:", np.round(get_state()[[0,1,2,3,4,5,7,8,9,10,11,12]], 2))

    if not args.keep_grippers:
        # Every training episode starts with both grippers OPEN (state ~1.3-1.4). Starting
        # closed (~6.3) puts the policy off-distribution from the first frame (training-box
        # finding 2026-09-13). Open them before the first observation is taken.
        print("== opening grippers to match dataset start state ==")
        q = get_state()
        g = np.array([np.clip((q[i] - GRIP_CAL[s][3]) / (GRIP_CAL[s][2] - GRIP_CAL[s][3]), 0, 1)
                      for i, s in ((6, "left"), (13, "right"))])  # 0=open, 1=closed
        for _ in range(45):
            if ESTOP_FLAG.exists():
                raise SystemExit("E-STOP during gripper open")
            g = np.maximum(g - MAX_GRIP_STEP, 0.0)
            left.command(np.concatenate([q[0:6], [1.0 - g[0]]]))
            right.command(np.concatenate([q[7:13], [1.0 - g[1]]]))
            time.sleep(1.0 / CTRL_HZ)
        time.sleep(0.3)
        print("   gripper state now:", np.round(get_state()[[6, 13]], 2), "(dataset start ~1.34-1.44)")
        _g = get_state()[[6, 13]]
        if np.any(_g < 1.0) or np.any(_g > 2.3):   # 2026-09-18: a re-referenced gripper read 7.55 here and the run went on with a closed hand
            raise SystemExit(f"ABORT: gripper state after opening is {np.round(_g, 2)} (expected ~1.3-1.7): gripper calibration / wrap is off, "
                             "or a gripper did not open — not starting the policy")

    ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

    def select_plan(chunks, snap):
        """Pick one of K sampled chunks. 'consistent': the sample closest to what the current plan would do over the
        wall-clock steps both cover (new step i <-> old step shift+i), starting at the expected install offset."""
        k = len(chunks)
        if k == 1 or args.plan_select == "first" or snap.get("prev_chunk") is None:
            return 0, [0.0] * k
        d = int(snap["delay"]); shift = snap["tick"] - snap["prev_obs_tick"]
        h = min(15, CHUNK - shift - d, CHUNK - d)
        if h < 3:
            return 0, [0.0] * k
        old = snap["prev_chunk"][shift + d: shift + d + h][:, ARM_IDX]
        costs = [float(np.abs(c[d:d + h][:, ARM_IDX] - old).mean()) for c in chunks]
        return int(np.argmin(costs)), costs

    def infer_worker(snap):
        t0 = time.perf_counter()
        try:
            if use_server:
                chunks, norms, _ = runner.infer_k(snap["frames"], snap["state"], args.task, args.plan_samples, rtc=snap.get("rtc"))
                sel, costs = select_plan(chunks, snap)
                norm, chunk = norms[sel], chunks[sel]
            else:
                norm, chunk = runner.infer(snap["frames"], snap["state"], args.task, snap["leftover"], snap["delay"])
                sel, costs = 0, [0.0]
            result_q.put({"norm": norm, "chunk": chunk, "snap_tick": snap["tick"], "sel": sel, "costs": costs,
                          "latency": time.perf_counter() - t0, "epoch": snap.get("epoch", 0)})
            if snap["tick"] > 0:
                save_frames(snap["frames"])  # JPEG encode AFTER the result is handed over (was ~10 ms on the latency path)
        except Exception as e:
            result_q.put({"error": repr(e), "epoch": snap.get("epoch", 0)})

    def snapshot(tick, leftover, delay):
        if synced is not None:
            synced.fetch()               # blocks until every camera has a frame newer than now (<= sync timeout)
        fr = {n: c.frame() for n, c in cams.items()}
        return {"frames": fr, "state": get_state() + OFF14, "tick": tick, "leftover": leftover, "delay": delay}

    try:
        # ---- initial chunk (synchronous) ----
        print("== first inference ==")
        _snap0 = snapshot(0, None, 0)
        save_frames(_snap0["frames"])
        norm_chunk, chunk = runner.infer(_snap0["frames"], get_state() + OFF14, args.task, None, 0)
        q_now = get_state() + OFF14  # policy frame
        arm_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        jump = np.abs(chunk[0] - q_now)[arm_idx].max()
        print(f"first-action jump: {jump:.3f} rad (limit {args.max_start_jump})")
        if jump > args.max_start_jump:
            raise SystemExit("ABORT: first action too far from current pose — "
                             "reposition arms near the training home pose.")

        # command vector layout per arm: [q1..q6, grip_norm]
        def split(a14):
            # policy gripper action: 0=OPEN, 1=CLOSED; i2rt normalized: 0=closed, 1=open
            l = np.concatenate([a14[0:6], [np.clip(1.0 - a14[6], 0, 1)]])
            r = np.concatenate([a14[7:13], [np.clip(1.0 - a14[13], 0, 1)]])
            return l, r

        state0 = get_state() + OFF14  # policy frame (physical + delta)
        # last commanded, in action space (gripper normalized)
        last_cmd = state0.copy()
        for gi, side in ((6, "left"), (13, "right")):
            _, _, tcl, top = GRIP_CAL[side]
            last_cmd[gi] = np.clip((last_cmd[gi] - top) / (tcl - top), 0, 1)  # 0=open, 1=closed

        park_cmd = last_cmd.copy()  # startup pose; arms return here on normal completion

        if args.simple:
            # ---------------- simplest possible scheme ----------------
            rate = np.array([args.max_arm_step] * 6 + [MAX_GRIP_STEP] +
                            [args.max_arm_step] * 6 + [MAX_GRIP_STEP])
            t_start = time.monotonic()
            tick = 0
            n_chunks = 0
            # ramp into chunk[0] of the first (already computed) chunk
            from prism_pi.control.sag_comp import compensate14
            sag = (lambda a, q: compensate14(a, q)) if not args.no_sag_comp else (lambda a, q: a)
            q_sag = get_state()
            for i in range(RAMP_TICKS):
                alpha = (i + 1) / RAMP_TICKS
                cmd = np.clip((1 - alpha) * park_cmd + alpha * sag(chunk[0], q_sag), a_lo, a_hi)
                cmd = last_cmd + np.clip(cmd - last_cmd, -rate, rate)
                l, r = split(cmd); left.command(l); right.command(r)
                last_cmd = cmd; log["cmds"].append(cmd.copy())
                time.sleep(1.0 / CTRL_HZ)
            estopped = False
            abort_excursion = False
            wr1_bad = 0
            while time.monotonic() - t_start < args.duration and not estopped and not abort_excursion:
                # fresh obs, synchronous predict (arm holds; robot-side thread feeds watchdog)
                t0 = time.perf_counter()
                _frames = {n: c.frame() for n, c in cams.items()}
                save_frames(_frames)
                _, chunk = runner.infer(_frames, get_state(), args.task, None, 0)
                log["latency"].append(time.perf_counter() - t0)
                log["chunks"].append(chunk)
                n_chunks += 1
                for i in range(min(args.commit_ticks, CHUNK)):
                    if ESTOP_FLAG.exists() or not (left.alive and right.alive):
                        print("E-STOP or chain down — stopping.")
                        estopped = True
                        break
                    if i % 10 == 0:
                        q_sag = get_state()
                        log["states"].append(q_sag)
                        # post-miss excursion signatures: wrist pitched up, or shoulder near max with the wrist
                        # not pointing down (arm extended toward the backdrop). Training grasps: lift <= 2.8,
                        # wr1 at grasp -1.25..-0.37.
                        excursion = (max(q_sag[3], q_sag[10]) > args.abort_wr1) or \
                                    (q_sag[1] > 2.6 and q_sag[3] > 0.0) or (q_sag[8] > 2.6 and q_sag[10] > 0.0)
                        if args.abort_wr1 > 0 and excursion:
                            wr1_bad += 1
                            if wr1_bad >= 2:  # two consecutive samples 1/3 s apart
                                print(f"ABORT: post-miss excursion (L lift/wr1 {q_sag[1]:.2f}/{q_sag[3]:+.2f}, "
                                      f"R {q_sag[8]:.2f}/{q_sag[10]:+.2f}) — parking.")
                                estopped = False; abort_excursion = True
                                break
                        else:
                            wr1_bad = 0
                    target = np.clip(sag(chunk[i], q_sag), a_lo, a_hi)
                    cmd = last_cmd + np.clip(target - last_cmd, -rate, rate)
                    l, r = split(cmd); left.command(l); right.command(r)
                    last_cmd = cmd; log["cmds"].append(cmd.copy())
                    tick += 1
                    HEARTBEAT.write_text(json.dumps({"t": time.time(), "mode": "dry" if args.dry else "LIVE",
                        "task": args.task, "tick": tick, "chunk_ptr": i, "switches": n_chunks,
                        "latency_ms": round(1000 * log["latency"][-1]), "delay_ticks": 0}))
                    time.sleep(1.0 / CTRL_HZ)
            clean_end = not estopped
            print(f"== simple done: {n_chunks} chunks, {tick} exec ticks, "
                  f"mean latency {np.mean(log['latency'])*1000:.0f} ms ==")
            if clean_end and not args.dry:
                print("== returning to park pose ==")
                while np.abs(park_cmd - last_cmd).max() > 0.01:
                    if ESTOP_FLAG.exists():
                        break
                    cmd = last_cmd + np.clip(park_cmd - last_cmd, -rate * 0.5, rate * 0.5)
                    l, r = split(cmd); left.command(l); right.command(r)
                    last_cmd = cmd
                    time.sleep(1.0 / CTRL_HZ)
                time.sleep(0.5)
                print("parked; releasing torque")
            return

        # ---------------- sim-matched scheme (default) ----------------
        # Mirrors the simulator's run_policy_sim.py: a fresh observation every `commit_ticks`, chunk step i executed at
        # obs_tick + i, the same rate limits and action envelope, a 15-tick ramp at the very start, no RTC and no
        # blending at chunk switches. The sim's inference is instantaneous; here it takes ~8-10 ticks, so instead of
        # holding the arm while it runs (the --simple dwell, which the sim ablation showed costs picks) the arm keeps
        # following the previous chunk and the new one is entered at step (tick - obs_tick): every executed step keeps
        # the wall-clock alignment it had in the sim, only the first ~9 steps of each chunk are never executed.
        from prism_pi.control.sag_comp import compensate14
        sag = (lambda a, q: compensate14(a, q)) if not args.no_sag_comp else (lambda a, q: a)
        rate = np.array([args.max_arm_step] * 6 + [MAX_GRIP_STEP] +
                        [args.max_arm_step] * 6 + [MAX_GRIP_STEP])
        arb = None
        if args.arbitrate:
            from prism_pi.control.arm_arbiter import ZoneArbiter
            arb = ZoneArbiter(zone_w=args.arb_zone, timeout_s=args.arb_timeout, hold_mode=args.arb_hold)
            print(f"== arm arbitration ON: one arm at a time within {args.arb_zone*100:.0f} cm of the centre line, "
                  f"dither timeout {args.arb_timeout:.0f} s ==")
        obs_tick = 0        # tick of the observation behind the current chunk (first chunk: at rest = tick 0)
        pre = shifter = None; pre_dir = np.zeros(3); pre_dq = {"left": np.zeros(6), "right": np.zeros(6)}
        arb_rejoin = {"left": False, "right": False}   # arm was held by the arbiter and has not yet caught up with the policy's plan
        down_cap = None
        if args.max_down_speed_cm_s[0] > 0:
            from prism_pi.control.tool_offset import DownSpeedCap, ToolShifter as _TS
            down_cap = DownSpeedCap(shifter if shifter is not None else _TS(), args.max_down_speed_cm_s[0], max(args.max_down_speed_cm_s[1], 1.0), max_step=args.max_arm_step)
            print(f"== downward fingertip speed cap: {args.max_down_speed_cm_s[0]:.0f} cm/s above 15 cm -> {args.max_down_speed_cm_s[1]:.0f} cm/s at <= 5 cm ==")
        _pv = np.array(args.preclose_advance_mm, float) / 1000.0
        if np.linalg.norm(_pv) > 0:
            from prism_pi.control.tool_offset import PreCloseAdvance, ToolShifter, floor_limited
            shifter = ToolShifter(); pre_dir = _pv / np.linalg.norm(_pv)
            import os as _os   # YAM_PRECLOSE_THR: test-only override of the close-onset threshold (offline harness)
            pre = PreCloseAdvance(float(np.linalg.norm(_pv)), args.preclose_ticks, relative=is_relative, close_thr=float(_os.environ.get("YAM_PRECLOSE_THR", 0.20)))
            print(f"== pre-close advance ON: {np.round(_pv * 1000, 1)} mm in the gripper frame over {args.preclose_ticks} ticks at each close onset "
                  f"(relative-action bookkeeping: {is_relative}); pad-centre floor {args.preclose_pad_floor_cm:.1f} cm ==")
        snap_tick = 0       # tick of the latest observation taken (pending or installed)
        pending = False
        delay_ema = 9.0     # inference latency in ticks (feeds the optional RTC prefix only)
        n_chunks = 1
        starved_warned = False
        wr1_bad = 0
        estopped = abort_excursion = False
        log["chunks"].append(chunk)
        log["infer_tick"].append(0); log["install_tick"].append(0); log["offsets"].append(0)
        tick = 0
        ctrl_path = Path(args.control_file) if args.control_file else None
        if ctrl_path is not None and not use_server:
            raise SystemExit("--control-file needs the policy server (--server on)")
        duration = args.duration if args.duration > 0 else float("inf")
        epoch = 0           # bumped at every resume: results of observations taken before a pause are discarded
        paused_s = 0.0

        def read_ctrl():
            try:
                return json.loads(ctrl_path.read_text())
            except Exception:
                return {}

        def beat(phase, ptr=0):
            HEARTBEAT.write_text(json.dumps({
                "t": time.time(), "mode": "dry" if args.dry else "LIVE", "phase": phase, "run_dir": str(run_dir),
                "task": args.task, "tick": tick, "chunk_ptr": int(ptr), "switches": n_chunks,
                "latency_ms": round(1000 * (log["latency"][-1] if log["latency"] else 0)),
                "delay_ticks": round(delay_ema, 1), "elapsed_s": round(tick / CTRL_HZ, 1),
                "duration_s": args.duration if args.duration > 0 else None, "paused_s": round(paused_s, 1),
                "ckpt": Path(args.ckpt).name,
            }))

        end_requested = False
        prev_sent = None      # last SENT command (for --command-substeps interpolation)
        segment = 0
        segments.append({"segment": 0, "task": args.task, "tick": 0, "t_wall": time.time()})
        t_start = time.monotonic()
        print(f"== running: sim-matched scheme, observe every {args.commit_ticks} ticks "
              f"({args.commit_ticks / CTRL_HZ:.2f} s), continuous execution{' + RTC' if args.rtc else ''} ==")
        while time.monotonic() - t_start < duration:
            if ESTOP_FLAG.exists() or not (left.alive and right.alive):
                print(f"E-STOP or chain down (left={left.alive}, right={right.alive}) — stopping commands.")
                estopped = True
                break

            if ctrl_path is not None and tick % 5 == 0:
                ctrl = read_ctrl()
                if ctrl.get("cmd") == "end":
                    print(f"== END requested at {tick / CTRL_HZ:.1f} s ==")
                    end_requested = True
                    break
                if ctrl.get("cmd") == "pause" and tick >= RAMP_TICKS:
                    # PAUSE: keep sending the last command (arms stay where they are, under power), no inference.
                    print(f"== PAUSED at {tick / CTRL_HZ:.1f} s: holding both arms in place ==")
                    t_pause = time.monotonic(); hold_sent = last_cmd - OFF14
                    if down_cap is not None:   # what was actually sent last (the down-speed cap may lag the policy-frame command)
                        hold_sent[0:6], hold_sent[7:13] = down_cap.last["left"], down_cap.last["right"]
                    hold_sent[[6, 13]] = log["grip_sent"][-1]
                    hl, hr = split(hold_sent); n_hold = 0
                    while True:
                        if ESTOP_FLAG.exists() or not (left.alive and right.alive):
                            break
                        if n_hold % 5 == 0:
                            ctrl = read_ctrl()
                            if ctrl.get("cmd") in ("run", "end"):
                                break
                        if n_hold % 15 == 0:
                            paused_s = time.monotonic() - t_pause; beat("paused")
                        left.command(hl); right.command(hr)
                        n_hold += 1
                        dt = t_pause + n_hold / CTRL_HZ - time.monotonic()
                        if dt > 0:
                            time.sleep(dt)
                    if ESTOP_FLAG.exists() or not (left.alive and right.alive):
                        continue      # handled at the top of the loop
                    if ctrl.get("cmd") == "end":
                        print("== END requested while paused ==")
                        end_requested = True
                        break
                    # RESUME: the running plan becomes "stay here"; an observation is taken right away and its chunk enters
                    # through the normal path (rate limit + RTC guidance toward the hold). Older results are discarded.
                    if ctrl.get("task") and ctrl["task"] != args.task:
                        print(f"== task changed: {args.task!r} -> {ctrl['task']!r} ==")
                        args.task = ctrl["task"]
                    epoch += 1
                    while not result_q.empty():
                        result_q.get()
                    pending = False
                    chunk = np.tile(last_cmd, (CHUNK, 1)); obs_tick = tick; snap_tick = tick - args.commit_ticks
                    wr1_bad = 0
                    t_start = time.monotonic() - tick / CTRL_HZ      # paused time does not count
                    paused_s = 0.0
                    segment += 1
                    segments.append({"segment": segment, "task": args.task, "tick": tick, "t_wall": time.time()})
                    print(f"== RESUMED after {time.monotonic() - t_pause:.1f} s (segment {segment}) ==")

            # install a finished chunk as soon as it lands, aligned to the tick its observation was taken at
            if not result_q.empty():
                res = result_q.get()
                if res.get("epoch", 0) != epoch:
                    continue          # observation from before a pause
                pending = False
                if "error" in res:
                    print(f"INFERENCE FAILED ({res['error']}) — stopping the policy and parking under torque.")
                    abort_excursion = True
                    break
                norm_chunk, chunk, obs_tick = res["norm"], res["chunk"], res["snap_tick"]
                log["plan_sel"].append(res.get("sel", 0)); log["plan_costs"].append(res.get("costs", [0.0]))
                offset = tick - obs_tick
                delay_ema = 0.7 * delay_ema + 0.3 * offset
                n_chunks += 1
                starved_warned = False
                log["chunks"].append(chunk); log["install_tick"].append(tick)
                log["offsets"].append(offset); log["latency"].append(res["latency"])

            # next observation on the sim's cadence (slips only if inference is slower than the cadence)
            if not pending and tick - snap_tick >= args.commit_ticks:
                snap_tick = tick
                leftover = norm_chunk[min(tick - obs_tick, CHUNK - 1):].clone() if (args.rtc and not use_server) else None
                extra = {"prev_chunk": chunk, "prev_obs_tick": obs_tick, "epoch": epoch}   # current plan, for plan selection
                if args.rtc and use_server:   # absolute tail of the running plan from this observation tick onward
                    extra["rtc"] = {"prev_abs": chunk[min(tick - obs_tick, CHUNK - 2):].copy(),
                                    "delay": int(round(delay_ema)) + 1, "horizon": args.rtc_horizon}
                if synced is not None:
                    # frame-synced: the wait for fresh frames (~7 ms) + decode must not stall the 30 Hz command loop, so the
                    # observation itself is taken in the worker; it lands within the same tick (labelled `tick`)
                    def _obs_and_infer(t=tick, lo=leftover, d=int(round(delay_ema)) + 1, ex=extra):
                        infer_worker({**snapshot(t, lo, d), **ex})
                    threading.Thread(target=_obs_and_infer, daemon=True).start()
                else:
                    snap = {**snapshot(tick, leftover, int(round(delay_ema)) + 1), **extra}
                    threading.Thread(target=infer_worker, args=(snap,), daemon=True).start()
                pending = True
                log["infer_tick"].append(tick)

            q_now = get_state()
            log["states"].append(q_now); log["t_wall"].append(time.time()); log["segment"].append(segment)
            if tick % 10 == 0 and args.abort_wr1 > 0:
                # post-miss excursion signatures: wrist pitched up, or shoulder near max with the wrist not
                # pointing down (arm extended toward the backdrop). Training grasps: lift <= 2.8, wr1 -1.25..-0.37.
                excursion = (max(q_now[3], q_now[10]) > args.abort_wr1) or \
                            (q_now[1] > 2.6 and q_now[3] > 0.0) or (q_now[8] > 2.6 and q_now[10] > 0.0)
                wr1_bad = wr1_bad + 1 if excursion else 0
                if wr1_bad >= 2:  # two consecutive samples 1/3 s apart
                    print(f"ABORT: post-miss excursion (L lift/wr1 {q_now[1]:.2f}/{q_now[3]:+.2f}, "
                          f"R {q_now[8]:.2f}/{q_now[10]:+.2f}) — parking.")
                    abort_excursion = True
                    break

            ptr = tick - obs_tick
            if ptr >= CHUNK:
                if not starved_warned:
                    print(f"tick {tick}: chunk exhausted (inference slower than {CHUNK} ticks?) — holding last step")
                    starved_warned = True
                ptr = CHUNK - 1
            target = np.clip(sag(chunk[ptr], q_now), a_lo, a_hi)
            rate_t = rate
            if arb is not None:   # policy frame in, policy frame out (OFF14 is zero unless the offset hypothesis is on)
                target, arb_info = arb.step(q_now + OFF14, target, lookahead=chunk[ptr:ptr + 25], prefer=args.arb_prefer)
                log["arb_owner"].append({"left": 1, "right": 2}.get(arb_info["owner"], 0))
                if arb_info["held"] is not None:   # a held arm retreats gently (homing rate), not at the policy's rate limit
                    o = 0 if arb_info["held"] == "left" else 7
                    rate_t = rate.copy(); rate_t[o:o + 6] = np.minimum(rate_t[o:o + 6], 0.03)
                # SOFT REJOIN (2026-09-19, after a crash): when a held arm is released its target snaps back to the policy's
                # plan, which can be far away (the right arm was 0.5 rad from it, mid-retreat) - at the normal 0.10 rad/tick
                # that was a 21 cm -> 7 cm plunge in 0.3 s. Keep the gentle rate until the command has caught up with the plan.
                for _side, _o in (("left", 0), ("right", 7)):
                    if arb_info["held"] == _side:
                        arb_rejoin[_side] = True
                    elif arb_rejoin[_side]:
                        if np.max(np.abs(target[_o:_o + 6] - last_cmd[_o:_o + 6])) < 0.06:
                            arb_rejoin[_side] = False
                        else:
                            rate_t = rate_t.copy(); rate_t[_o:_o + 6] = np.minimum(rate_t[_o:_o + 6], 0.03)
            if tick < RAMP_TICKS:  # ramp from the start pose into the plan, exactly as the sim does
                alpha = (tick + 1) / RAMP_TICKS
                target = (1 - alpha) * last_cmd + alpha * target
            cmd = last_cmd + np.clip(target - last_cmd, -rate_t, rate_t)
            sent = cmd - OFF14  # physical = policy frame - delta
            if pre is not None:   # one-shot tool-frame advance at close onset (see tool_offset.py)
                held, adv = pre.step(tick, cmd, {"left": obs_tick, "right": obs_tick})
                for side, o in (("left", 0), ("right", 7)):
                    dq = np.zeros(6)
                    if adv[side] > 1e-6:
                        q_true = sent[o:o + 6].copy()
                        d_tool = floor_limited(shifter, side, q_true, pre_dir * adv[side], args.preclose_pad_floor_cm / 100.0)
                        dq = shifter.shift(side, q_true, d_tool) - q_true
                    # the shift-induced joint change is itself rate limited (0.04 rad/tick) and kept inside the envelope
                    dq = pre_dq[side] + np.clip(dq - pre_dq[side], -0.04, 0.04); pre_dq[side] = dq
                    sent[o:o + 6] = np.clip(sent[o:o + 6] + dq, a_lo[o:o + 6] - OFF14[o:o + 6], a_hi[o:o + 6] - OFF14[o:o + 6])
                    sent[o + 6] = min(sent[o + 6], held[side])
                log["tool_shift_mm"].append((1000 * adv["left"], 1000 * adv["right"]))
            if args.grip_assist > 0:  # squeeze harder than the policy asks once it is clearly closing (pads only touch otherwise)
                for o in (6, 13):
                    g_pol = min(cmd[o], sent[o])   # respects a gripper held by the pre-close advance
                    sent[o] = min(1.0, g_pol + args.grip_assist * float(np.clip((g_pol - 0.10) / 0.10, 0, 1)))
            if down_cap is not None:   # final stage: never command the fingertips downward faster than the cap
                sent[0:6] = down_cap.apply("left", sent[0:6]); sent[7:13] = down_cap.apply("right", sent[7:13])
            if args.command_substeps > 1 and prev_sent is not None:
                # SMOOTH COMMANDS: instead of one position step per 30 Hz tick (a zero-order hold the arm's PD loop answers with a
                # jerk every 33 ms - the left wrist camera was found oscillating at exactly 30 Hz, ~9 px, 2026-09-22), ramp the
                # sent target linearly from the previous one to this one over the tick period in N sub-steps. The full target is
                # reached at the END of the tick (one tick of extra command lag, like a first-order hold).
                N = args.command_substeps
                for k in range(1, N + 1):
                    sub = prev_sent + (sent - prev_sent) * (k / N)
                    l, r = split(sub); left.command(l); right.command(r)
                    dt = t_start + (tick + k / N) / CTRL_HZ - time.monotonic()
                    if dt > 0:
                        time.sleep(dt)
            else:
                l, r = split(sent); left.command(l); right.command(r)
            prev_sent = sent.copy()
            last_cmd = cmd
            log["cmds"].append(cmd.copy()); log["grip_sent"].append(sent[[6, 13]].copy()); log["sent_arm"].append(sent[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].copy())

            if tick % 15 == 0:
                beat("running", ptr)

            tick += 1
            dt = t_start + tick / CTRL_HZ - time.monotonic()
            if dt > 0:
                time.sleep(dt)

        clean_end = not estopped
        print(f"== done: {tick} ticks, {n_chunks} chunks, mean latency {np.mean(log['latency'] or [0])*1000:.0f} ms, "
              f"install offset mean {np.mean(log['offsets'][1:] or [0]):.1f} / max {max(log['offsets'][1:] or [0])} ticks ==")
        if synced is not None and synced.stats["wait_ms"]:
            w, a = np.array(synced.stats["wait_ms"]), np.array(synced.stats["age_ms"])
            print(f"== frame sync: wait median {np.median(w):.1f} / max {w.max():.1f} ms, frame age at state read median {np.median(a):.1f} / max {a.max():.1f} ms ==")

        # ---- return to park pose, then zero torque (skip after e-stop) ----
        if clean_end and not args.dry:
            print("== returning to park pose ==")
            HOME_STEP = 0.05  # rad/tick, gentler than the run itself
            hrate = np.array([HOME_STEP] * 6 + [MAX_GRIP_STEP] + [HOME_STEP] * 6 + [MAX_GRIP_STEP])
            htick = 0
            t_home = time.monotonic()
            while np.abs(park_cmd - last_cmd).max() > 0.01 and htick < 12 * CTRL_HZ:
                if ESTOP_FLAG.exists():
                    print("E-STOP during homing — stopping.")
                    break
                cmd = last_cmd + np.clip(park_cmd - last_cmd, -hrate, hrate)
                lcmd, rcmd = split(cmd - OFF14)
                left.command(lcmd)
                right.command(rcmd)
                last_cmd = cmd
                htick += 1
                if htick % 15 == 0:
                    beat("parking")
                dt = t_home + htick / CTRL_HZ - time.monotonic()
                if dt > 0:
                    time.sleep(dt)
            time.sleep(0.5)  # settle at park before torque release
            print("parked; releasing torque")
    finally:
        _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)  # let cleanup finish
        try:
            np.savez(run_dir / "run.npz",
                     cmds=np.array(log["cmds"]) if log["cmds"] else np.zeros(0),
                     chunks=np.array(log["chunks"]) if log["chunks"] else np.zeros(0),
                     offsets=np.array(log["offsets"]), latency=np.array(log["latency"]),
                     states=np.array(log["states"]) if log["states"] else np.zeros(0),
                     infer_tick=np.array(log["infer_tick"]), install_tick=np.array(log["install_tick"]),
                     scheme="simple" if args.simple else "sim_matched", task=args.task,
                     plan_samples=args.plan_samples, plan_sel=np.array(log["plan_sel"]), rtc=bool(args.rtc), rtc_horizon=args.rtc_horizon,
                     arbitrate=bool(args.arbitrate), arb_owner=np.array(log["arb_owner"], dtype=np.int8), grip_assist=float(args.grip_assist), grip_sent=np.array(log["grip_sent"]), preclose_advance_mm=np.array(args.preclose_advance_mm, float), tool_shift_mm=np.array(log["tool_shift_mm"]), sent_arm=np.array(log["sent_arm"]), down_cap_ticks=np.array([down_cap.n_capped["left"], down_cap.n_capped["right"]] if down_cap is not None else [0, 0]),
                     plan_costs=np.array([np.pad(np.asarray(c, float), (0, max(0, args.plan_samples - len(c)))) for c in log["plan_costs"]]) if log["plan_costs"] else np.zeros(0),
                     joint_offset_vec_deg=np.array(args.joint_offset_vec_deg if args.joint_offset_vec_deg else [0.0]*6),
                     t_wall=np.array(log["t_wall"], dtype=np.float64), segment=np.array(log["segment"], dtype=np.int32),
                     segments_json=json.dumps(segments), ckpt=str(args.ckpt), dry=bool(args.dry), argv=json.dumps(_sys_argv))
            print(f"log saved to {run_dir}/run.npz")
        except Exception as e:
            print("log save failed:", e)
        stop.set()
        time.sleep(0.2)
        for a in (left, right):
            try:
                a.close()
            except Exception:
                pass
        HEARTBEAT.unlink(missing_ok=True)
        if use_server:
            runner.close()
        print("torques zeroed")


if __name__ == "__main__":
    main()

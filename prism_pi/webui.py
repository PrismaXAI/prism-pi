"""YAM station control WebUI: camera previews, policy rollout sessions, software e-stop.

Serves on 0.0.0.0:8080 — open http://<machine-ip>:8080 from any device on the LAN/Wi-Fi (mouse or touch / iPad).

- Live MJPEG streams of all three cameras (policy-view framing: full sensor -> 640x400).
- Rollout sessions: pick a checkpoint (listed in config/checkpoints.json) and one of its tasks, then either a single TIMED run
  or a CONTINUOUS session. The UI spawns the executor (prism_pi.control.executor; sim-matched scheme, RTC + arbitration, policy server) and
  steers it through a control file: PAUSE = both arms held in place under power with the policy stopped (reset the
  scene, optionally change the task), RESUME = fresh observation and carry on, END SESSION = park both arms under
  power, then release. The policy server keeps the model in VRAM; it is only asked to load when the selected
  checkpoint differs from the one it holds (and it is started here if it is not running).
- E-STOP: kills any robot-control processes (policy executor, dry-run, teleop), then sends motor-off to all 7 motors
  on both CAN buses, and latches a flag file that executors check before commanding. RESET clears the latch (does NOT
  re-enable anything by itself).
- /frame/{cam}.jpg gives the latest frame to other local consumers (the executor shares the cameras through it).
- LeRobot dataset option: the session's frames are recorded at the full capture rate and, when the session ends, paired
  with the executor's per-tick log (state, commanded action, wall clock) by prism_pi.rollout_to_lerobot into
  datasets/<name>/ (LeRobot v3, the training dataset's exact schema; one episode per run segment, i.e. PAUSE/RESUME =
  scene reset = new episode). Conversions run one at a time in the background; progress shows in the UI.
- POST /session/start with "dry": true runs the executor's offline harness (stub arms, no CAN, real cameras + policy
  server) instead of the robot: pipeline tests without touching the arms (same as YAM_UI_TEST=1 for every session).

Run: setsid nohup .venv/bin/python -m prism_pi.webui > webui.log 2>&1 &   (or ./rig_up.sh)
"""

import ctypes
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _die_with_parent():
    """PR_SET_PDEATHSIG: kernel kills the child if this process dies, so ffmpeg
    can never be orphaned holding a camera device."""
    _libc.prctl(1, signal.SIGKILL)

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from prism_pi.paths import CHECKPOINTS, CONTROL, DATASETS, ESTOP_FLAG, HEARTBEAT, ROOT, RUNS
from prism_pi.policy import client as policy_client
from prism_pi.rig import mapping as rig_mapping

PY = sys.executable      # the executor / policy server / dataset converter run in this UI's own environment
# camera name -> /dev/v4l/by-path device, from config/rig_mapping.json (all three cameras share one USB serial: address by port)
CAMS = {n: rig_mapping.camera_device(n) for n in ("high", "left_wrist", "right_wrist")}
STREAM_FPS = 15          # browser streams (the capture pipeline itself runs at CAP_FPS)
# camera capture rate: higher = fresher frames (sampling delay 1/fps), more decode CPU. Default from config/rig_mapping.json
# ("capture_fps", default 120); two cameras behind ONE hub do not get USB bandwidth for 120 fps each: use 60 there.
CAP_FPS = int(os.environ.get("YAM_CAP_FPS", rig_mapping.load().get("capture_fps", 120)))
# processes the e-stop kills (never this UI itself, never the policy server)
# (matched against full command lines by `pkill -f`: this repo's arm-driving modules and tools, plus the i2rt SDK's teleop scripts)
KILL_PATTERNS = ("prism_pi.control.executor|prism_pi.control.offline_test|prism_pi.rig.dryrun"
                 "|probe_tip_height|probe_cam_latency|replay_dataset_actions|identify_arms|calibrate_gripper"
                 "|minimum_gello|bimanual_lead_follower|motor_chain_robot.py|record_replay")

# the standard pick-and-place rig command (2026-09-19): sim-matched scheme + RTC via the server + arm arbitration (one arm
# at a time near the centre line) + post-miss excursion guard 0.6 rad. Folding is bimanual and pitches the wrists up
# legitimately (dataset wrist_1 up to +1.5 rad), so its checkpoints run WITHOUT arbitration and WITHOUT the guard.
FLAG_PRESETS = {"pnp": ["--rtc", "--arbitrate", "--server", "on", "--abort-wr1", "0.6"],
                "bimanual": ["--rtc", "--server", "on", "--abort-wr1", "0"]}
PNP_FLAGS = FLAG_PRESETS["pnp"]


def load_ckpts() -> dict:
    """config/checkpoints.json (see checkpoints.example.json there): key -> {"label", "path" (checkpoint or shim dir,
    absolute or relative to the repository root), "tasks" (verbatim training strings), "flags" (a preset name from
    FLAG_PRESETS or an explicit list of executor arguments; default "pnp")}."""
    f = CHECKPOINTS
    if not f.exists():
        print(f"[webui] no config/{f.name}: no checkpoints to roll out (copy config/checkpoints.example.json)", flush=True)
        return {}
    out = {}
    for key, c in json.loads(f.read_text()).items():
        flags = c.get("flags", "pnp")
        out[key] = {"label": c.get("label", key), "path": (ROOT / c["path"]).resolve(), "tasks": list(c["tasks"]),
                    "flags": list(FLAG_PRESETS[flags]) if isinstance(flags, str) else [str(x) for x in flags]}
    return out


CKPTS = load_ckpts()   # key -> label, checkpoint dir, tasks, executor flags
DURATIONS = [30, 60, 90, 150]
GRIP_ASSIST = 0.05   # extra closure (normalized 0..1) added once the policy is clearly closing; ramps in over cmd 0.10-0.20
# test hook (no CAN access): YAM_UI_TEST=1 runs the executor's offline harness (stub arms, --dry) instead of the real thing
TEST_DRY = os.environ.get("YAM_UI_TEST") == "1"
PORT = int(os.environ.get("YAM_UI_PORT", 8080))
# DATASETS: LeRobot datasets written by prism_pi.rollout_to_lerobot: datasets/<name>/{meta,data,videos}
DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
DEFAULT_DATASET = "rollouts"

app = FastAPI()


class CamThread(threading.Thread):
    """One ffmpeg subprocess per camera (decode/scale/encode off the GIL);
    this thread only splits the concatenated-JPEG stream and keeps the latest frame."""

    def __init__(self, name: str, device: str):
        super().__init__(daemon=True)
        self.cam_name, self.device = name, device
        self.jpeg: bytes | None = None
        self.fps = 0.0
        self.count = 0
        self.t_frame = 0.0
        self.ok = False
        self.new = threading.Condition()   # notified on every decoded frame (event-driven streaming, no polling delay)
        self.start()

    def run(self):
        placeholder = np.zeros((400, 640, 3), np.uint8)
        cv2.putText(placeholder, f"{self.cam_name}: NO SIGNAL", (140, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 220), 2)
        self.jpeg = cv2.imencode(".jpg", placeholder)[1].tobytes()
        # full camera rate and no demuxer/muxer buffering: the executor samples the newest frame at observation time,
        # so at 30 fps its frame is at most 33 ms old (was up to 67 ms at 15 fps); measured pipeline delay ~8 ms
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay",
               "-f", "v4l2", "-input_format", "mjpeg",
               "-video_size", "1920x1200", "-framerate", str(CAP_FPS),
               "-i", self.device,
               "-vf", "scale=640:400", "-fps_mode", "passthrough",   # no padding: the camera delivers ~119.65 fps, cfr would duplicate a frame every ~2.9 s
               "-c:v", "mjpeg", "-q:v", "6", "-f", "mjpeg", "-flush_packets", "1", "pipe:1"]
        while True:
            # bufsize=0: read() returns whatever the pipe holds. With the default buffered pipe, read(65536) blocked until
            # 64 KB had accumulated = 6-7 JPEGs -> frames reached memory in ~200 ms bursts (430 ms at the old 15 fps),
            # i.e. up to that much extra camera latency for the executor (found 2026-09-21).
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                                    preexec_fn=_die_with_parent)
            buf = b""
            n, t0 = 0, time.time()
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    soi = buf.find(b"\xff\xd8")
                    if soi < 0:
                        break
                    eoi = buf.find(b"\xff\xd9", soi + 2)
                    if eoi < 0:
                        break
                    self.jpeg = buf[soi:eoi + 2]
                    buf = buf[eoi + 2:]
                    self.ok = True
                    self.count += 1
                    self.t_frame = time.time()
                    recorder.on_frame(self.cam_name, self.jpeg, self.t_frame, self.count)
                    with self.new:
                        self.new.notify_all()
                    n += 1
                    if n % 24 == 0:
                        now = time.time()
                        self.fps = 24 / (now - t0)
                        t0 = now
                if len(buf) > 4 << 20:  # corrupted stream guard
                    buf = b""
            self.ok = False
            proc.kill()
            time.sleep(2)  # device gone or busy; retry


class Recorder:
    """Debug hook: records every pipeline frame of all three cameras (the raw JPEGs, no re-encode, full capture rate)
    into runs/rec_<timestamp>/<cam>.mjpeg + <cam>.idx (t, count per frame) + meta.json. ~3.6 MB/s at 120 fps."""
    MAX_S = 900

    def __init__(self):
        self.lock = threading.Lock(); self.dir = None; self.files = {}; self.idx = {}; self.t0 = 0.0; self.n = 0

    @property
    def active(self):
        return self.dir is not None

    def start(self, tag: str = ""):
        with self.lock:
            if self.dir is not None:
                return self.dir
            self.dir = RUNS / f"rec_{time.strftime('%Y%m%d_%H%M%S')}"; self.dir.mkdir(parents=True, exist_ok=True)
            self.files = {n: open(self.dir / f"{n}.mjpeg", "ab") for n in CAMS}; self.idx = {n: [] for n in CAMS}
            self.t0 = time.time(); self.n = 0
            (self.dir / "meta.json").write_text(json.dumps({"t0": self.t0, "tag": tag, "cap_fps": CAP_FPS}))
            print(f"[recorder] START {self.dir.name}", flush=True)
            return self.dir

    def on_frame(self, cam: str, jpeg: bytes, t: float, count: int):
        if self.dir is None:
            return
        f = self.files.get(cam)
        if f is None:
            return
        try:
            off = f.tell(); f.write(jpeg); self.idx[cam].append((t, count, off, len(jpeg))); self.n += 1
        except Exception:
            pass
        if t - self.t0 > self.MAX_S:
            threading.Thread(target=self.stop, daemon=True).start()

    def stop(self):
        with self.lock:
            if self.dir is None:
                return
            d = self.dir; self.dir = None
            for n, f in self.files.items():
                try:
                    f.close()
                except Exception:
                    pass
                np.save(d / f"{n}.idx.npy", np.array(self.idx[n], dtype=np.float64))
            meta = json.loads((d / "meta.json").read_text()); meta.update({"t1": time.time(), "frames": self.n}); (d / "meta.json").write_text(json.dumps(meta))
            print(f"[recorder] STOP {d.name}: {self.n} frames, {(time.time() - self.t0):.0f} s", flush=True)
            self.files = {}


recorder = Recorder()
cams = {n: CamThread(n, p) for n, p in CAMS.items()}


def disable_all_motors() -> list[str]:
    """Kill controller processes, then motor-off every motor on both buses."""
    log = []
    r = subprocess.run(["pkill", "-f", KILL_PATTERNS], capture_output=True)
    log.append(f"pkill controllers: rc={r.returncode} (0=killed some, 1=none running)")
    time.sleep(0.3)  # let killed processes release sockets
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface

    for ch in ("can0", "can1"):
        try:
            iface = DMSingleMotorCanInterface(channel=ch, bustype="socketcan")
            off = []
            for mid in range(1, 8):
                try:
                    iface.motor_off(mid)
                    off.append(mid)
                except Exception:
                    pass
            iface.close()
            log.append(f"{ch}: motor_off sent to {off}")
        except Exception as e:
            log.append(f"{ch}: FAILED to open bus: {e}")
    return log


def read_heartbeat():
    try:
        data = json.loads(HEARTBEAT.read_text())
        if time.time() - data.get("t", 0) < 3:
            return data
    except Exception:
        pass
    return None


class Session:
    """One executor process at a time, steered through CONTROL. All robot motion stays in the executor process."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.info: dict = {}            # mode, ckpt key, task, duration of the current / last session
        self.requested = "run"          # last command written to CONTROL
        self.busy: str | None = None    # "server" (starting the policy server) | "loading" (checkpoint swap) | None
        self.message = ""               # last refusal / failure, shown in the UI
        self.finished = False           # this session's post-run work (stop recording, queue the conversion) is done
        self.convert_q: queue.Queue = queue.Queue()   # dataset conversions, run one at a time (they append to the same dataset)
        self.convert: dict = {}         # state of the current / last conversion, shown in the UI
        threading.Thread(target=self._convert_loop, daemon=True).start()

    # -- helpers --
    def _write_ctrl(self, cmd: str, task: str | None = None):
        tmp = CONTROL.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cmd": cmd, "task": task, "t": time.time()}))
        os.replace(tmp, CONTROL)
        self.requested = cmd

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def active(self) -> bool:
        return self.alive() or self.busy is not None or read_heartbeat() is not None

    def log_tail(self, n: int = 7) -> list[str]:
        if not self.log_path or not self.log_path.exists():
            return []
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2); f.seek(max(0, f.tell() - 6000))
                lines = f.read().decode(errors="replace").splitlines()
        except Exception:
            return []
        keep = [ln for ln in lines if ln.strip() and not ln.startswith((" ", "WARNING", "ERROR:root", "Traceback", "Exception in thread",
                                                                        "DM Error", "ValueError: file descriptor"))]
        return [ln[:160] for ln in keep[-n:]]

    # -- actions --
    def start(self, ckpt_key: str, task: str, mode: str, duration: int, grip_assist: bool = False, frame_sync: bool = False, record: bool = False, smooth: bool = False,
              dataset: str | None = None, dry: bool = False) -> str | None:
        """Returns a refusal reason, or None when the start sequence has begun. dataset = name of the LeRobot dataset the
        session's episodes are appended to (implies frame recording); dry = offline harness instead of the robot."""
        dataset = (dataset or "").strip() or None
        if dataset is not None and not DATASET_NAME_RE.match(dataset):
            return "Dataset name: letters, digits, '_', '.', '-' only (max 64)."
        with self.lock:
            if ESTOP_FLAG.exists():
                return "E-stop is latched: reset the latch first."
            if self.active():
                return "A session is already active."
            ck = CKPTS.get(ckpt_key)
            if ck is None or task not in ck["tasks"]:
                return "Unknown checkpoint / task."
            if not (ck["path"] / "model.safetensors").exists():
                return f"Checkpoint files not found: {ck['path'].name}"
            if mode not in ("timed", "continuous"):
                return "Unknown mode."
            self.info = {"mode": mode, "ckpt": ckpt_key, "task": task, "duration_s": int(duration) if mode == "timed" else None,
                         "grip_assist": bool(grip_assist), "frame_sync": bool(frame_sync), "record": bool(record or dataset is not None), "smooth": bool(smooth),
                         "dataset": dataset, "dry": bool(dry or TEST_DRY), "t_start": time.time()}
            self.message = ""
            self.busy = "server"
            self.log_path = None
            self.finished = False
        threading.Thread(target=self._start_worker, args=(ck, task, mode, int(duration), bool(grip_assist), bool(frame_sync)), daemon=True).start()
        return None

    def _start_worker(self, ck: dict, task: str, mode: str, duration: int, grip_assist: bool, frame_sync: bool):
        try:
            want = str(ck["path"].resolve())
            info = policy_client.server_info()
            if info is not None and (info.get("ckpt") != want or info.get("dtype") != "bf16"):
                # checkpoint swap = restart the server process with the new checkpoint (an in-process hot-swap leaves
                # CUDA-graph / TensorRT memory behind and OOMs on this 24 GB card; a fresh process is ~65 s either way)
                self.busy = "loading"
                try:
                    policy_client.shutdown()
                except Exception:
                    pass
                t0 = time.time()
                while policy_client.server_info() is not None:
                    if time.time() - t0 > 30:
                        raise RuntimeError("policy server did not stop for the checkpoint swap")
                    time.sleep(1)
                info = None
            if info is None:    # no policy server: start one with the wanted checkpoint already loaded
                self.busy = self.busy or "server"
                with open(ROOT / "policy_server.log", "ab") as lf:
                    subprocess.Popen([PY, "-u", "-m", "prism_pi.policy.server", "--ckpt", want], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                     env={**os.environ, "HF_HUB_OFFLINE": "1"}, start_new_session=True)
                t0 = time.time()
                while (info := policy_client.server_info()) is None or info.get("ckpt") is None:
                    if time.time() - t0 > 300:
                        raise RuntimeError("policy server did not come up within 300 s (see policy_server.log)")
                    time.sleep(2)
            if info.get("ckpt") != want:
                raise RuntimeError(f"policy server holds {info.get('ckpt')}, wanted {want}")
            if ESTOP_FLAG.exists():
                raise RuntimeError("e-stop engaged while preparing")
            with self.lock:
                self._write_ctrl("run", task)
                HEARTBEAT.unlink(missing_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                self.log_path = RUNS / f"ui_{stamp}_{mode}.log"
                self.log_path.parent.mkdir(exist_ok=True)
                out_dir = RUNS if not self.info.get("dry") else Path("/tmp/yam_ui_test")
                script = ["-m", "prism_pi.control.executor"] if not self.info.get("dry") else ["-m", "prism_pi.control.offline_test", "--dry", "--out", str(out_dir)]
                cmd = [PY, "-u", *script, "--task", task, "--duration", str(duration if mode == "timed" else 0),
                       "--ckpt", want, *ck.get("flags", PNP_FLAGS), "--control-file", str(CONTROL), "--run-name", f"ui_{stamp}"]
                self.info["run_dir"] = str(out_dir / f"ui_{stamp}")
                if grip_assist:
                    cmd += ["--grip-assist", str(GRIP_ASSIST)]
                if frame_sync:
                    cmd += ["--sync-obs-to-frame"]
                if self.info.get("smooth"):
                    cmd += ["--command-substeps", "4"]
                if self.info.get("record"):
                    rd = recorder.start(tag=self.log_path.name); self.info["rec_dir"] = rd.name
                with open(self.log_path, "ab") as lf:
                    lf.write((" ".join(cmd) + "\n").encode())
                    self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                                 env={**os.environ, "HF_HUB_OFFLINE": "1"}, start_new_session=True)
                threading.Thread(target=self._watch_worker, args=(self.proc, dict(self.info)), daemon=True).start()
        except Exception as e:
            self.message = f"Start failed: {e}"
            if self.info.get("record") and recorder.active and self.info.get("rec_dir") == recorder.dir.name:
                recorder.stop()
        finally:
            self.busy = None

    def _watch_worker(self, proc: subprocess.Popen, info: dict):
        """Waits for the executor to exit, then closes the session's recording and queues the dataset conversion
        (does not depend on a browser polling /status)."""
        proc.wait()
        self._finish(info)

    def _finish(self, info: dict):
        with self.lock:
            if self.finished or info.get("t_start") != self.info.get("t_start"):
                return
            self.finished = True
        if recorder.active and info.get("rec_dir") == recorder.dir.name:
            recorder.stop()
        if info.get("dataset") and info.get("run_dir") and info.get("rec_dir"):
            job = {"dataset": info["dataset"], "run_dir": info["run_dir"], "rec_dir": str(RUNS / info["rec_dir"]),
                   "meta": {"ckpt_key": info.get("ckpt"), "mode": info.get("mode"), "ui_log": self.log_path.name if self.log_path else None,
                            "grip_assist": info.get("grip_assist"), "frame_sync": info.get("frame_sync"), "smooth": info.get("smooth")},
                   "queued": time.time()}
            self.convert_q.put(job)
            self.convert = {"state": "queued", "dataset": job["dataset"], "t": time.time(), "tail": "", "log_file": None, "episodes": None}

    def _convert_loop(self):
        while True:
            job = self.convert_q.get()
            log_path = RUNS / f"convert_{time.strftime('%Y%m%d_%H%M%S')}_{job['dataset']}.log"
            self.convert = {"state": "running", "dataset": job["dataset"], "t": time.time(), "tail": "", "log_file": log_path.name, "episodes": None}
            print(f"[dataset] converting {Path(job['run_dir']).name} + {Path(job['rec_dir']).name} -> datasets/{job['dataset']}", flush=True)
            try:
                if not (Path(job["run_dir"]) / "run.npz").exists():
                    raise RuntimeError(f"no run.npz in {job['run_dir']} (executor did not get to its log?)")
                if not (Path(job["rec_dir"]) / "meta.json").exists():
                    raise RuntimeError(f"no frame recording at {job['rec_dir']}")
                cmd = [PY, "-u", "-m", "prism_pi.rollout_to_lerobot", "--run", job["run_dir"], "--rec", job["rec_dir"],
                       "--root", str(DATASETS / job["dataset"]), "--meta", json.dumps(job["meta"])]
                with open(log_path, "wb") as lf:
                    lf.write((" ".join(cmd) + "\n").encode())
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                            env={**os.environ, "HF_HUB_OFFLINE": "1"}, start_new_session=True)
                while proc.poll() is None:
                    self.convert["tail"] = _tail_line(log_path)
                    time.sleep(1)
                self.convert["tail"] = _tail_line(log_path)
                if proc.returncode != 0:
                    raise RuntimeError(f"rollout_to_lerobot exit {proc.returncode}: {self.convert['tail']}")
                self.convert.update({"state": "done", "episodes": dataset_info(job["dataset"]).get("episodes")})
                print(f"[dataset] done: {self.convert['tail']}", flush=True)
            except Exception as e:
                self.convert.update({"state": "failed", "tail": str(e)})
                print(f"[dataset] FAILED: {e}", flush=True)

    def command(self, cmd: str, task: str | None = None) -> str | None:
        with self.lock:
            if not (self.alive() or read_heartbeat() is not None):
                return "No session is running."
            if cmd == "run" and task:
                ck = CKPTS.get(self.info.get("ckpt", ""), None)
                if ck is not None and task not in ck["tasks"]:
                    return "That task does not belong to the loaded checkpoint."
                self.info["task"] = task
            self._write_ctrl(cmd, task if cmd == "run" else None)
        return None

    def status(self) -> dict:
        hb = read_heartbeat()
        estop = ESTOP_FLAG.exists()
        alive = self.alive()
        if self.proc is not None and not alive and hb is None and not self.finished and self.busy is None:   # session over (fallback to the watcher)
            self._finish(dict(self.info))
        if estop:
            state = "estop"
        elif self.busy:
            state = self.busy
        elif hb is not None and hb.get("phase") in ("running", "paused", "parking"):
            state = "pausing" if (hb["phase"] == "running" and self.requested == "pause") else hb["phase"]
        elif alive:
            state = "parking" if self.requested == "end" else "starting"
        else:
            state = "idle"
        return {"state": state, "info": self.info, "requested": self.requested, "message": self.message,
                "exit_code": None if (self.proc is None or alive) else self.proc.returncode,
                "log": self.log_tail(), "log_file": self.log_path.name if self.log_path else None,
                "recording": recorder.dir.name if recorder.active else None, "rec_dir": self.info.get("rec_dir"),
                "convert": self.convert, "convert_pending": self.convert_q.qsize()}


def _tail_line(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 4000))
            lines = [ln for ln in f.read().decode(errors="replace").splitlines() if re.match(r"\[\d\d:\d\d:\d\d\] ", ln)]   # the converter's own lines
        return lines[-1][:200] if lines else ""
    except Exception:
        return ""


def dataset_info(name: str) -> dict:
    """Episode / frame counts of datasets/<name> straight from its info.json (no LeRobot import in the UI process)."""
    p = DATASETS / name / "meta" / "info.json"
    try:
        info = json.loads(p.read_text())
        return {"name": name, "exists": True, "episodes": info.get("total_episodes", 0), "frames": info.get("total_frames", 0)}
    except Exception:
        return {"name": name, "exists": False, "episodes": 0, "frames": 0}


def list_datasets() -> list[dict]:
    out = []
    if DATASETS.exists():
        for d in sorted(DATASETS.iterdir()):
            if (d / "meta" / "info.json").exists():
                out.append(dataset_info(d.name))
    return out


session = Session()


class StartReq(BaseModel):
    ckpt: str
    task: str
    mode: str = "timed"
    duration: int = 60
    grip_assist: bool = False
    frame_sync: bool = False
    record: bool = False
    smooth: bool = False
    dataset: str | None = None   # LeRobot dataset name (datasets/<name>); None = no dataset
    dry: bool = False            # offline harness (stub arms, no CAN): pipeline tests only


class ResumeReq(BaseModel):
    task: str | None = None


def _reply(err: str | None, what: str = ""):
    print(f"[{time.strftime('%H:%M:%S')}] {what}: {err or 'ok'}", flush=True)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=409)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/stream/{cam}")
def stream(cam: str):
    c = cams[cam]

    def gen():
        # every 1/STREAM_FPS s send the NEWEST frame (waiting for a fresh one if none arrived since the last send);
        # a slow client skips frames instead of drifting behind
        t_last, last_count = 0.0, -1
        while True:
            wait = t_last + 1.0 / STREAM_FPS - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            with c.new:
                c.new.wait_for(lambda: c.count != last_count, timeout=1.0)
            buf, last_count = c.jpeg, c.count
            t_last = time.monotonic()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n")

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/frame/{cam}.jpg")
def frame(cam: str):
    return Response(cams[cam].jpeg, media_type="image/jpeg")


@app.get("/frames")
def frames(fresh: int = 0, timeout_ms: int = 20):
    """All three cameras in one reply (JSON, base64 JPEGs + capture times). fresh=1: wait (up to timeout_ms) until
    every camera has delivered a frame captured AFTER this request arrived, so the executor's observation is taken on
    fresh, mutually aligned images (--sync-obs-to-frame) instead of whatever was newest at its tick."""
    import base64
    t0 = time.time(); deadline = t0 + timeout_ms / 1000
    out = {}
    for n, c in cams.items():
        if fresh:
            with c.new:
                c.new.wait_for(lambda: c.t_frame > t0, timeout=max(0.0, deadline - time.time()))
        out[n] = {"t": c.t_frame, "count": c.count, "jpeg": base64.b64encode(c.jpeg).decode()}
    return {"t_req": t0, "t_reply": time.time(), "cams": out}


@app.post("/session/start")
def session_start(req: StartReq):
    return _reply(session.start(req.ckpt, req.task, req.mode, req.duration, req.grip_assist, req.frame_sync, req.record, req.smooth, req.dataset, req.dry),
                  f"start {req.mode} {req.ckpt} {req.task!r} assist={req.grip_assist} frame_sync={req.frame_sync} record={req.record} smooth={req.smooth}"
                  f" dataset={req.dataset!r}{' DRY' if req.dry else ''}")


@app.post("/session/pause")
def session_pause():
    return _reply(session.command("pause"), "pause")


@app.post("/session/resume")
def session_resume(req: ResumeReq):
    return _reply(session.command("run", req.task), f"resume {req.task!r}")


@app.post("/session/end")
def session_end():
    return _reply(session.command("end"), "end")


@app.post("/record/start")
def record_start():
    """Manual recording (outside a session), e.g. during teleop; stops with /record/stop, the e-stop, or after 15 min."""
    return {"ok": True, "rec": recorder.start(tag="manual").name}


@app.post("/record/stop")
def record_stop():
    name = recorder.dir.name if recorder.active else None
    recorder.stop()
    return {"ok": True, "rec": name}


@app.post("/estop")
def estop():
    ESTOP_FLAG.write_text(str(time.time()))
    log = disable_all_motors()
    recorder.stop()
    return JSONResponse({"engaged": True, "log": log})


@app.post("/reset")
def reset():
    ESTOP_FLAG.unlink(missing_ok=True)
    return JSONResponse({"engaged": False})


@app.get("/config")
def config():
    return {"ckpts": [{"key": k, "label": v["label"], "tasks": v["tasks"], "available": (v["path"] / "model.safetensors").exists()}
                      for k, v in CKPTS.items()], "durations": DURATIONS, "grip_assist": GRIP_ASSIST, "default_dataset": DEFAULT_DATASET}


@app.get("/datasets")
def datasets(name: str | None = None):
    """Existing LeRobot datasets (datasets/<name>), or the counts of one name (also when it does not exist yet)."""
    if name:
        return dataset_info(name) if DATASET_NAME_RE.match(name) else {"name": name, "exists": False, "episodes": 0, "frames": 0, "error": "bad name"}
    return {"datasets": list_datasets()}


_srv_cache = {"t": 0.0, "v": None}


def _server_info_cached():
    if time.time() - _srv_cache["t"] > 2.0:
        try:
            _srv_cache["v"] = policy_client.server_info()
        except Exception:
            _srv_cache["v"] = None
        _srv_cache["t"] = time.time()
    return _srv_cache["v"]


@app.get("/status")
def status():
    can = {}
    for ch in ("can0", "can1", "can_left", "can_right"):
        p = Path(f"/sys/class/net/{ch}/operstate")
        if p.exists():
            can[ch] = p.read_text().strip()
    srv = _server_info_cached()
    loaded = None
    if srv and srv.get("ckpt"):
        loaded = next((k for k, v in CKPTS.items() if str(v["path"].resolve()) == srv["ckpt"]), Path(srv["ckpt"]).name)
    return {
        "estop": ESTOP_FLAG.exists(),
        "executor": read_heartbeat(),
        "session": session.status(),
        "server": {"up": srv is not None, "loaded": loaded},
        "stream_fps": STREAM_FPS,
        "can": can,
        "cams": {n: {"ok": c.ok, "fps": round(c.fps, 1), "frames": c.count, "age_ms": round((time.time() - c.t_frame) * 1000) if c.t_frame else None} for n, c in cams.items()},
        "t": time.time(),
    }


PAGE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>YAM station</title>
<style>
:root { --bg:#0F1216; --panel:#171B21; --raised:#1F252D; --line:#2A313B; --ink:#ECEDE9; --mut:#8D96A2;
        --ok:#3FA873; --ok-d:#2A7350; --bad:#D9483F; --bad-d:#8A2C27; --warn:#E2A93B; --warn-d:#94701F; --info:#4C8FD6; }
* { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
html,body { height:100%; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-text-size-adjust:100%; }
main { max-width:1280px; margin:0 auto; padding:12px 16px calc(132px + env(safe-area-inset-bottom)); }
header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
h1 { font-size:17px; font-weight:650; margin:0; }
#state { font:700 13px/1 ui-monospace, Menlo, monospace; padding:8px 12px; border-radius:6px; letter-spacing:.04em;
  background:var(--raised); color:var(--mut); }
#state.live { background:#173527; color:#6FD3A0; } #state.hold { background:#3B2F14; color:var(--warn); }
#state.stop { background:#42201E; color:#FF8A82; } #state.busy { background:#16283D; color:#8DBBEB; }
#sys { color:var(--mut); font:12px/1.4 ui-monospace, Menlo, monospace; margin-left:auto; text-align:right; }
.cams { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:8px; }
.cam { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.cam img { width:100%; aspect-ratio:8/5; display:block; background:#000; }
.cam .lbl { display:flex; justify-content:space-between; padding:6px 10px; font:12px/1 ui-monospace, Menlo, monospace; color:var(--mut); }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; margin-top:10px; }
.grid { display:grid; grid-template-columns:1.1fr 1.4fr 1fr; gap:12px; }
label.f { display:block; font-size:12px; color:var(--mut); text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
select { width:100%; min-height:56px; font-size:17px; color:var(--ink); background:var(--raised); border:1px solid var(--line);
  border-radius:10px; padding:0 40px 0 14px; appearance:none; -webkit-appearance:none;
  background-image:linear-gradient(45deg, transparent 50%, var(--mut) 50%), linear-gradient(135deg, var(--mut) 50%, transparent 50%);
  background-position:calc(100% - 22px) 25px, calc(100% - 15px) 25px; background-size:7px 7px; background-repeat:no-repeat; }
select:disabled { opacity:.45; }
.seg { display:flex; background:var(--raised); border:1px solid var(--line); border-radius:10px; padding:4px; gap:4px; min-height:56px; }
.seg button { flex:1; background:transparent; color:var(--mut); font-size:16px; font-weight:650; border-radius:7px; min-height:46px; }
.seg button.on { background:#2E3844; color:var(--ink); }
.seg button:disabled { opacity:.45; }
#row2 { margin-top:12px; }
#assist small, #fsync small, #rec small, #smooth small, #ds small { display:block; font-size:12px; font-weight:500; color:var(--mut); pointer-events:none; }
#assist.on small, #fsync.on small, #smooth.on small { color:#6FD3A0; } #rec.on small, #ds.on small { color:#FF8A82; }
input[type=text] { width:100%; min-height:56px; font-size:17px; color:var(--ink); background:var(--raised); border:1px solid var(--line);
  border-radius:10px; padding:0 14px; font-family:ui-monospace, Menlo, monospace; }
input[type=text]:disabled { opacity:.45; }
#dsinfo { font:12px/1.4 ui-monospace, Menlo, monospace; color:var(--mut); margin-top:6px; min-height:17px; }
#dsinfo.busy { color:#8DBBEB; } #dsinfo.bad { color:#FF8A82; }
button { border:none; border-radius:12px; cursor:pointer; font-weight:750; font-family:inherit; touch-action:manipulation;
  user-select:none; -webkit-user-select:none; }
button:disabled { cursor:default; }
.actions { display:grid; grid-template-columns:2fr 1fr; gap:12px; margin-top:14px; }
.big { min-height:84px; font-size:24px; letter-spacing:.03em; color:#fff; }
.big small { display:block; pointer-events:none; font-size:12.5px; font-weight:500; letter-spacing:0; opacity:.85; margin-top:3px; }
#primary.go { background:var(--ok); box-shadow:0 4px 0 var(--ok-d); }
#primary.pause { background:var(--warn); color:#1E1706; box-shadow:0 4px 0 var(--warn-d); }
#primary.wait { background:var(--raised); color:var(--mut); box-shadow:none; }
#endbtn { background:#33404F; box-shadow:0 4px 0 #222B36; }
#endbtn:disabled, #primary:disabled { opacity:.4; box-shadow:none; }
.big:not(:disabled):active { transform:translateY(3px); box-shadow:none !important; }
#prog { height:8px; background:var(--raised); border-radius:4px; margin-top:14px; overflow:hidden; }
#prog i { display:block; height:100%; width:0; background:var(--ok); transition:width .9s linear; }
#prog.hold i { background:var(--warn); }
#runline { display:flex; justify-content:space-between; gap:12px; margin-top:8px; color:var(--mut);
  font:13px/1.4 ui-monospace, Menlo, monospace; font-variant-numeric:tabular-nums; }
#msg { margin-top:10px; color:#FFB4AE; font-size:14px; min-height:0; }
details { margin-top:10px; color:var(--mut); } summary { cursor:pointer; padding:8px 0; font-size:13px; }
#log { font:12px/1.5 ui-monospace, Menlo, monospace; white-space:pre-wrap; word-break:break-word; }
#stopbar { position:fixed; left:0; right:0; bottom:0; display:flex; gap:10px; align-items:stretch;
  padding:10px 16px calc(10px + env(safe-area-inset-bottom)); background:linear-gradient(transparent, var(--bg) 22%); }
#stopbtn { flex:1; background:var(--bad); color:#fff; font-size:34px; letter-spacing:.08em; min-height:96px; box-shadow:0 5px 0 var(--bad-d); }
#stopbtn:active { transform:translateY(4px); box-shadow:0 1px 0 var(--bad-d); }
#resetbtn { background:var(--panel); color:var(--mut); border:1px solid var(--line); font-size:14px; padding:0 20px; min-width:120px; }
#resetbtn.need { color:var(--ink); border-color:var(--bad); }
@media (max-width:820px) { .cams { grid-template-columns:1fr 1fr; } .cams .cam:first-child { grid-column:1 / -1; }
  .grid { grid-template-columns:1fr; } .actions { grid-template-columns:1fr; } #sys { display:none; } }
</style></head><body><main>
<header>
  <h1>YAM station</h1>
  <span id="state">…</span>
  <span id="sys"></span>
</header>
<div class="cams">
  <div class="cam"><img data-cam="high" src="/stream/high" alt="overhead camera"><div class="lbl"><span>high (overhead)</span><span id="fps-high"></span></div></div>
  <div class="cam"><img data-cam="left_wrist" src="/stream/left_wrist" alt="left wrist camera"><div class="lbl"><span>left_wrist</span><span id="fps-left_wrist"></span></div></div>
  <div class="cam"><img data-cam="right_wrist" src="/stream/right_wrist" alt="right wrist camera"><div class="lbl"><span>right_wrist</span><span id="fps-right_wrist"></span></div></div>
</div>
<section class="panel">
  <div class="grid">
    <div><label class="f" for="ckpt">Checkpoint</label><select id="ckpt"></select></div>
    <div><label class="f" for="task">Task</label><select id="task"></select></div>
    <div><label class="f">Mode</label>
      <div class="seg" id="mode"><button data-mode="timed" class="on">Timed run</button><button data-mode="continuous">Continuous</button></div>
    </div>
  </div>
  <div class="grid" id="row2">
    <div><label class="f">Options</label>
      <div class="seg" id="opts"><button id="assist" data-on="0">Grip assist<small>off</small></button><button id="fsync" data-on="0">Frame-synced obs<small>off</small></button><button id="rec" data-on="0">Record frames<small>off</small></button><button id="smooth" data-on="0">Smooth commands<small>off</small></button><button id="ds" data-on="0">LeRobot dataset<small>off</small></button></div>
    </div>
    <div id="dswrap"><label class="f" for="dsname">Dataset (datasets/&lt;name&gt;)</label><input type="text" id="dsname" autocapitalize="off" autocorrect="off" spellcheck="false" maxlength="64" placeholder="rollouts"><div id="dsinfo"></div></div>
    <div id="durwrap"><label class="f">Duration</label><div class="seg" id="dur"></div></div>
  </div>
  <div class="actions">
    <button id="primary" class="big go">START</button>
    <button id="endbtn" class="big" disabled>END SESSION<small>park both arms under power</small></button>
  </div>
  <div id="prog"><i></i></div>
  <div id="runline"><span id="runl"></span><span id="runr"></span></div>
  <div id="msg"></div>
  <details><summary>Executor log</summary><div id="log"></div></details>
</section>
</main>
<div id="stopbar">
  <button id="stopbtn">E-STOP</button>
  <button id="resetbtn">Reset latch</button>
</div>
<script>
const $ = id => document.getElementById(id);
let cfg = null, mode = 'timed', dur = 60, assist = false, fsync = false, rec = false, smooth = false, ds = false, st = null, sending = false;

async function post(url, body) {
  sending = true;
  try {
    const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body || {})});
    const j = await r.json().catch(() => ({}));
    $('msg').textContent = r.ok ? '' : (j.error || 'request failed');
    return j;
  } catch (e) { $('msg').textContent = 'No connection to the station.'; }
  finally { sending = false; poll(true); }
}

function fillTasks() {
  const ck = cfg.ckpts.find(c => c.key === $('ckpt').value);
  if (!ck) { $('task').innerHTML = ''; return; }
  const cur = $('task').value;
  $('task').innerHTML = ck.tasks.map(t => `<option>${t}</option>`).join('');
  if (ck.tasks.includes(cur)) $('task').value = cur;
}
function setSeg(el, attr, val) { el.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset[attr] === String(val))); }

async function init() {
  cfg = await (await fetch('/config')).json();
  $('ckpt').innerHTML = cfg.ckpts.map(c => `<option value="${c.key}" ${c.available ? '' : 'disabled'}>${c.label}${c.available ? '' : ' — not on disk'}</option>`).join('')
    || '<option value="" disabled selected>no checkpoints.json</option>';
  $('dur').innerHTML = cfg.durations.map(d => `<button data-dur="${d}">${d} s</button>`).join('');
  setSeg($('dur'), 'dur', dur);
  try { const s = JSON.parse(localStorage.getItem('yam-ui') || '{}');
    if (s.ckpt && cfg.ckpts.some(c => c.key === s.ckpt && c.available)) $('ckpt').value = s.ckpt;
    fillTasks(); if (s.task) $('task').value = s.task; if ($('task').selectedIndex < 0) $('task').selectedIndex = 0;
    if (s.mode) mode = s.mode; if (s.dur) dur = s.dur; if (s.assist) assist = true; if (s.fsync) fsync = true; if (s.smooth) smooth = true;
    if (s.ds) ds = true; if (s.dsname) $('dsname').value = s.dsname; } catch (e) { fillTasks(); }
  if (!$('dsname').value) $('dsname').value = cfg.default_dataset || 'rollouts';
  setAssist(assist); setFsync(fsync); setRec(rec); setSmooth(smooth); setDs(ds);
  setSeg($('mode'), 'mode', mode); setSeg($('dur'), 'dur', dur);
  poll(); dsInfo();
}
function remember() { try { localStorage.setItem('yam-ui', JSON.stringify({ckpt:$('ckpt').value, task:$('task').value, mode, dur, assist, fsync, smooth, ds, dsname:$('dsname').value})); } catch (e) {} }
function setDs(v) { ds = v; const b = $('ds'); b.classList.toggle('on', v);
  b.innerHTML = 'LeRobot dataset<small>' + (v ? 'one episode per segment, 30 fps' : 'off') + '</small>';
  if (v && !rec) setRec(true); }
$('ds').onclick = () => { if ($('ds').disabled) return; setDs(!ds); remember(); dsInfo(); };
let dsTimer = null, dsLast = null;
async function dsInfo() {
  const name = $('dsname').value.trim();
  if (!name) { dsLast = null; return; }
  try { dsLast = await (await fetch('/datasets?name=' + encodeURIComponent(name))).json(); } catch (e) { dsLast = null; }
  render();
}
$('dsname').oninput = () => { remember(); clearTimeout(dsTimer); dsTimer = setTimeout(dsInfo, 400); };
function setSmooth(v) { smooth = v; const b = $('smooth'); b.classList.toggle('on', v);
  b.innerHTML = 'Smooth commands<small>' + (v ? '120 Hz interpolated targets' : 'off') + '</small>'; }
$('smooth').onclick = () => { if ($('smooth').disabled) return; setSmooth(!smooth); remember(); };
function setRec(v) { rec = v; const b = $('rec'); b.classList.toggle('on', v);
  b.innerHTML = 'Record frames<small>' + (v ? 'all cameras, full rate, this session' : 'off') + '</small>'; }
$('rec').onclick = () => { if ($('rec').disabled) return; setRec(!rec); };
function setFsync(v) { fsync = v; const b = $('fsync'); b.classList.toggle('on', v);
  b.innerHTML = 'Frame-synced obs<small>' + (v ? 'wait for fresh frames' : 'off') + '</small>'; }
$('fsync').onclick = () => { if ($('fsync').disabled) return; setFsync(!fsync); remember(); };
function setAssist(v) { assist = v; const b = $('assist'); b.classList.toggle('on', v);
  b.innerHTML = 'Grip assist<small>' + (v ? '+' + cfg.grip_assist + ' closure when closing' : 'off') + '</small>'; }
$('assist').onclick = () => { if ($('assist').disabled) return; setAssist(!assist); remember(); };

$('ckpt').onchange = () => { fillTasks(); remember(); };
$('task').onchange = remember;
$('mode').onclick = e => { const b = e.target.closest('button'); if (!b || b.disabled) return; mode = b.dataset.mode; setSeg($('mode'), 'mode', mode); remember(); render(); };
$('dur').onclick = e => { const b = e.target.closest('button'); if (!b || b.disabled) return; dur = +b.dataset.dur; setSeg($('dur'), 'dur', dur); remember(); };

$('primary').onclick = () => {
  const s = st && st.session.state;
  if (s === 'idle') post('/session/start', {ckpt:$('ckpt').value, task:$('task').value, mode, duration:dur, grip_assist:assist, frame_sync:fsync, record:rec, smooth:smooth,
                                            dataset: ds ? $('dsname').value.trim() : null});
  else if (s === 'running') post('/session/pause');
  else if (s === 'paused') post('/session/resume', {task:$('task').value});
};
$('endbtn').onclick = () => post('/session/end');
$('stopbtn').onclick = async () => {
  $('msg').textContent = 'Engaging e-stop…';
  const j = await post('/estop');
  if (j && j.log) { $('msg').textContent = 'E-stop engaged.'; $('log').textContent = j.log.join('\\n'); }
};
$('resetbtn').onclick = async () => {
  if (!confirm('Clear the e-stop latch? (does not re-enable motors)')) return;
  await post('/reset');
};

const LABEL = {idle:'IDLE', server:'STARTING POLICY SERVER', loading:'LOADING CHECKPOINT', starting:'STARTING · HOMING', running:'RUNNING',
  pausing:'PAUSING', paused:'PAUSED · ARMS HELD', parking:'PARKING', estop:'E-STOP LATCHED'};
const CLS = {idle:'', server:'busy', loading:'busy', starting:'busy', running:'live', pausing:'hold', paused:'hold', parking:'busy', estop:'stop'};
const fmt = s => { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };

function render() {
  if (!st || !cfg) return;
  const ses = st.session, s = ses.state, hb = st.executor, info = ses.info || {};
  $('state').textContent = LABEL[s] || s; $('state').className = CLS[s] || '';
  const idle = s === 'idle', paused = s === 'paused', active = !idle && s !== 'estop';
  if (active && info.ckpt) {            // show what is actually running
    if ($('ckpt').value !== info.ckpt) { $('ckpt').value = info.ckpt; fillTasks(); }
    if (!paused && info.task && document.activeElement !== $('task')) $('task').value = info.task;
    if (info.mode && info.mode !== mode) { mode = info.mode; setSeg($('mode'), 'mode', mode); }
    if (info.grip_assist !== undefined && info.grip_assist !== assist) setAssist(info.grip_assist);
    if (info.frame_sync !== undefined && info.frame_sync !== fsync) setFsync(info.frame_sync);
    if (info.smooth !== undefined && info.smooth !== smooth) setSmooth(info.smooth);
    if (info.dataset !== undefined && (info.dataset !== null) !== ds) setDs(info.dataset !== null);
    if (info.dataset && $('dsname').value !== info.dataset) { $('dsname').value = info.dataset; dsInfo(); }
  }
  $('ckpt').disabled = !idle;
  $('task').disabled = !(idle || paused);
  document.querySelectorAll('#mode button, #dur button, #assist, #fsync, #rec, #smooth, #ds').forEach(b => b.disabled = !idle);
  $('dsname').disabled = !idle;
  // dataset line: counts of the named dataset + the state of the background conversion
  const cv = ses.convert || {}, di = $('dsinfo'); let dtxt = '', dcls = '';
  if (dsLast && dsLast.name === $('dsname').value.trim()) dtxt = dsLast.exists ? `${dsLast.name}: ${dsLast.episodes} episodes · ${dsLast.frames} frames` : `${dsLast.name}: new dataset`;
  const strip = t => (t || '').replace(/^\\[[^\\]]*\\] /, '');   // drop the converter's [hh:mm:ss] prefix
  if (cv.state === 'queued' || cv.state === 'running') { dtxt = `converting → ${cv.dataset}` + (ses.convert_pending ? ` (+${ses.convert_pending} queued)` : '') + (cv.tail ? '\\n' + strip(cv.tail) : ''); dcls = 'busy'; }
  else if (cv.state === 'failed') { dtxt = `conversion FAILED → ${cv.dataset}: ${cv.tail}` + (cv.log_file ? ` (runs/${cv.log_file})` : ''); dcls = 'bad'; }
  else if (cv.state === 'done') { dtxt = strip(cv.tail) + (dtxt ? '\\n' + dtxt : ''); }
  if (cv.state === 'done' && cv.t !== window._dsDone) { window._dsDone = cv.t; dsInfo(); }
  di.textContent = dtxt; di.className = dcls; di.style.whiteSpace = 'pre-line';
  $('durwrap').style.visibility = mode === 'timed' ? '' : 'hidden';
  // the button is only rebuilt when its content changes: replacing its children every poll can swallow a tap
  // (no click is delivered when the pressed node disappears between touch-down and touch-up)
  let cls, dis = false, html;
  if (idle) { cls = 'big go'; html = mode === 'timed' ? `START ${dur} s RUN` : 'START CONTINUOUS';
    if (st.server.loaded !== $('ckpt').value) html += `<small>${st.server.up ? 'restarts the policy server with it (~1 min)' : 'starts the policy server first (~1 min)'}</small>`;
  } else if (s === 'running') { cls = 'big pause'; html = 'PAUSE<small>hold the arms in place to reset the scene</small>'; }
  else if (paused) { cls = 'big go'; html = 'RESUME<small>hands clear of the workspace</small>'; }
  else { cls = 'big wait'; dis = true; html = s === 'estop' ? 'E-STOP LATCHED' : (LABEL[s] || s) + '…'; }
  const p = $('primary'), key = cls + '|' + dis + '|' + html;
  if (p.dataset.key !== key) { p.dataset.key = key; p.className = cls; p.disabled = dis; p.innerHTML = html; }
  $('endbtn').disabled = !(s === 'running' || paused || s === 'pausing');
  $('resetbtn').classList.toggle('need', st.estop);

  const bar = $('prog'), fill = bar.firstElementChild;
  bar.className = paused ? 'hold' : '';
  if (hb && (s === 'running' || paused || s === 'pausing')) {
    if (hb.duration_s) { fill.style.width = Math.min(100, 100 * hb.elapsed_s / hb.duration_s) + '%';
      $('runl').textContent = fmt(hb.elapsed_s) + ' / ' + fmt(hb.duration_s); }
    else { fill.style.width = '100%'; $('runl').textContent = 'running ' + fmt(hb.elapsed_s); }
    if (paused) $('runl').textContent += '   paused ' + fmt(hb.paused_s);
    $('runr').textContent = hb.latency_ms + ' ms · chunk ' + hb.switches + (ses.recording ? ' · ● REC ' + ses.recording : '');
  } else { fill.style.width = '0'; $('runl').textContent = idle && ses.exit_code !== null && ses.log_file ? 'last session ended (exit ' + ses.exit_code + ') · ' + ses.log_file + (ses.rec_dir ? ' · recorded: ' + ses.rec_dir : '') : '';
    $('runr').textContent = ''; }
  if (ses.message && !$('msg').textContent) $('msg').textContent = ses.message;
  if (ses.log && ses.log.length) $('log').textContent = ses.log.join('\\n');

  $('sys').textContent = 'policy server: ' + (st.server.up ? (st.server.loaded || 'empty') : 'down') + '\\n' +
    Object.entries(st.can).map(([k, v]) => k + ':' + v).join('  ');
  for (const [n, c] of Object.entries(st.cams)) $('fps-' + n).textContent = c.ok ? Math.round(c.fps) + ' fps capture · ' + (st.stream_fps || 15) + ' fps view' : 'OFFLINE';
}

let timer = null;
async function poll(now) {
  if (now && timer) { clearTimeout(timer); timer = null; } else if (now === true && !timer) { /* fallthrough */ }
  try { st = await (await fetch('/status')).json(); render(); }
  catch (e) { $('state').textContent = 'NO CONNECTION'; $('state').className = 'stop'; }
  if (timer) clearTimeout(timer);
  timer = setTimeout(poll, 1000);
}
// MJPEG streams stall when a tab is backgrounded (iPad): restart them when it comes back or on error
function restartStreams() { document.querySelectorAll('img[data-cam]').forEach(i => i.src = '/stream/' + i.dataset.cam + '?t=' + Date.now()); }
document.addEventListener('visibilitychange', () => { if (!document.hidden) { restartStreams(); poll(true); } });
document.querySelectorAll('img[data-cam]').forEach(i => i.onerror = () => setTimeout(() => i.src = '/stream/' + i.dataset.cam + '?t=' + Date.now(), 1500));
init();
</script>
</body></html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

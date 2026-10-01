"""Turn a policy rollout (yam_executor.py run + the WebUI's frame recording) into LeRobot v3 episodes.

The executor logs every 30 Hz tick (run.npz: state read, commanded target, wall-clock time, segment id); the WebUI's
Recorder keeps every camera frame of the session at the full capture rate (runs/rec_<ts>/: raw JPEGs + capture times).
This script pairs each tick with the frame captured closest to its state read (<= half a capture period, ~4 ms at
120 fps) and appends one episode per run SEGMENT (segment boundaries = PAUSE/RESUME in the WebUI, i.e. scene resets)
to a LeRobot dataset with EXACTLY our training dataset's schema (codebase v3.0, 30 fps):

  observation.images.{high,left_wrist,right_wrist}  video 400x640x3, h264 yuv420p
  observation.state     float32[14]  measured joints (rad) + gripper in the TRAINING raw convention (open ~1.3, closed ~6.4)
  observation.velocity  float32[14]  finite difference of the state (state[t+1]-state[t]) * fps (the training data's
                                     velocity is encoder-derived; this is the closest thing a rollout has)
  action                float32[14]  the executor's rate-limited commanded target in the PHYSICAL frame (policy-frame
                                     command minus --joint-offset-vec-deg), gripper normalized 0 = open .. 1 = closed
                                     (the training dataset's action convention). Rig-side assists that were added AFTER
                                     this target (grip assist, pre-close advance, down-speed cap) are in run.npz, not here.
  task                  the segment's task string (verbatim training prompt)

Per-episode provenance goes to <root>/meta/rollouts.jsonl (run dir, recording, segment, checkpoint, frame ages, ...);
a run+segment already listed there is skipped, so re-running on the same inputs is safe.

  .venv/bin/python rollout_to_lerobot.py --run runs/<stamp> --rec runs/rec_<stamp> --root datasets/rollouts_fruits
  .venv/bin/python rollout_to_lerobot.py info --root datasets/rollouts_fruits      # episode / frame counts
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FPS = 30
CAMS = ("high", "left_wrist", "right_wrist")
H, W = 400, 640
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"]
NAMES = [f"{side}_{j}" for side in ("left", "right") for j in JOINTS]
FEATURES = {
    **{f"observation.images.{c}": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]} for c in CAMS},
    "observation.state": {"dtype": "float32", "shape": (14,), "names": NAMES},
    "observation.velocity": {"dtype": "float32", "shape": (14,), "names": NAMES},
    "action": {"dtype": "float32", "shape": (14,), "names": NAMES},
}
MAX_FRAME_AGE_S = 0.050   # a tick whose nearest frame is further away than this is a recording gap: the episode is refused
CRF = 23                  # x264 quality: ~1.5 Mbit/s at 640x400/30 fps like the training videos (LeRobot's default 30 gives ~0.6)
MIN_TICKS = 30            # segments shorter than 1 s are not worth an episode


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Recording:
    """One camera of a WebUI frame recording: capture times + memory-mapped JPEG bytes."""

    def __init__(self, rec_dir: Path, cam: str):
        self.idx = np.load(rec_dir / f"{cam}.idx.npy")          # (n, 4): t, count, offset, length
        if self.idx.ndim != 2 or len(self.idx) == 0:
            raise RuntimeError(f"{rec_dir.name}: no frames recorded for {cam}")
        self.t = self.idx[:, 0]
        self.data = np.memmap(rec_dir / f"{cam}.mjpeg", dtype=np.uint8, mode="r")

    def nearest(self, t: np.ndarray):
        """Index of the frame captured closest to each time in t, and the |dt| in seconds."""
        j = np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1)
        left_closer = (t - self.t[j - 1]) <= (self.t[j] - t)
        j = j - left_closer.astype(int)
        return j, np.abs(self.t[j] - t)

    def frame(self, j: int) -> np.ndarray:
        import cv2
        _, _, off, n = self.idx[j]
        img = cv2.imdecode(np.frombuffer(self.data[int(off):int(off) + int(n)], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"undecodable frame {j}")
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def load_run(run_dir: Path) -> dict:
    z = np.load(run_dir / "run.npz", allow_pickle=True)
    need = ("states", "cmds", "t_wall", "segment", "segments_json")
    missing = [k for k in need if k not in z.files]
    if missing:
        raise RuntimeError(f"{run_dir}: run.npz lacks {missing} (executor older than the per-tick log, or --simple scheme)")
    states, cmds, t_wall, seg = z["states"], z["cmds"], z["t_wall"], z["segment"]
    n = min(len(states), len(cmds), len(t_wall), len(seg))
    if n == 0:
        raise RuntimeError(f"{run_dir}: empty run")
    if not (len(states) == len(cmds) == len(t_wall) == len(seg)):
        log(f"WARNING: per-tick arrays differ in length (states {len(states)}, cmds {len(cmds)}, t_wall {len(t_wall)}, "
            f"segment {len(seg)}); using the first {n} ticks")
    off14 = np.zeros(14)
    if "joint_offset_vec_deg" in z.files:
        d6 = np.radians(np.asarray(z["joint_offset_vec_deg"], float))
        off14[0:6] = d6; off14[7:13] = d6
    return {"states": states[:n].astype(np.float64), "cmds": cmds[:n].astype(np.float64), "t_wall": t_wall[:n].astype(np.float64),
            "segment": seg[:n].astype(int), "segments": json.loads(str(z["segments_json"])), "off14": off14,
            "task": str(z["task"]) if "task" in z.files else "", "ckpt": str(z["ckpt"]) if "ckpt" in z.files else "",
            "dry": bool(z["dry"]) if "dry" in z.files else None}


def open_dataset(root: Path, repo_id: str, vcodec: str):
    """Create or resume the dataset. Frames are streamed straight into the video encoders (no PNG round trip); the
    encoder queue is deep enough that decoding JPEGs never outruns x264 at this resolution."""
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    av.logging.set_libav_level(av.logging.ERROR)   # x264 / mp4 muxer chatter goes straight to stderr otherwise (verified 2026-09-23)
    kw = dict(root=root, vcodec=vcodec, streaming_encoding=True, encoder_queue_maxsize=900)
    if (root / "meta" / "info.json").exists():
        info = json.loads((root / "meta" / "info.json").read_text())
        have = {k: list(v.get("shape")) for k, v in info["features"].items() if k in FEATURES}
        want = {k: list(v["shape"]) for k, v in FEATURES.items()}
        if have != want:
            raise RuntimeError(f"{root}: existing dataset features {have} differ from {want}")
        ds = LeRobotDataset.resume(repo_id, **kw)
        n0 = info["total_episodes"]
    else:
        ds = LeRobotDataset.create(repo_id, FPS, FEATURES, robot_type="yam", **kw)
        n0 = 0
    enc = getattr(ds.writer, "_streaming_encoder", None)
    if enc is not None and hasattr(enc, "crf"):
        enc.crf = CRF
    return ds, n0


def dropped_frames(ds) -> int:
    """Frames the streaming encoder dropped in the current episode (its queue ran full): must be 0 before save."""
    enc = getattr(ds.writer, "_streaming_encoder", None)
    return int(sum(getattr(enc, "_dropped_frames", {}).values())) if enc is not None else 0


def already_imported(root: Path) -> set:
    done = set()
    p = root / "meta" / "rollouts.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line); done.add((r["run"], int(r["segment"])))
            except Exception:
                pass
    return done


def convert(run_dir: Path, rec_dir: Path, root: Path, repo_id: str, meta: dict, vcodec: str, min_ticks: int, force: bool) -> int:
    run = load_run(run_dir)
    rec_meta = json.loads((rec_dir / "meta.json").read_text())
    recs = {c: Recording(rec_dir, c) for c in CAMS}
    run_key = str(run_dir.resolve())
    done = set() if force else already_imported(root)
    seg_task = {int(s["segment"]): s["task"] for s in run["segments"]}

    # per-tick nearest frames for every camera; the ticks a segment keeps must have a close frame on ALL cameras
    near = {c: recs[c].nearest(run["t_wall"]) for c in CAMS}
    age = np.max(np.stack([near[c][1] for c in CAMS]), axis=0)
    state = run["states"]
    action = run["cmds"] - run["off14"]                         # physical frame; gripper columns unaffected (off14 is 0 there)
    action[:, [6, 13]] = np.clip(action[:, [6, 13]], 0.0, 1.0)
    vel = np.zeros_like(state)
    vel[:-1] = (state[1:] - state[:-1]) * FPS
    if len(state) > 1:
        vel[-1] = vel[-2]

    plan = []
    for s in sorted(set(run["segment"].tolist())):
        idx = np.flatnonzero(run["segment"] == s)
        if (run_key, s) in done:
            log(f"segment {s}: already in {root.name} (meta/rollouts.jsonl), skipping"); continue
        if len(idx) < min_ticks:
            log(f"segment {s}: only {len(idx)} ticks (< {min_ticks}), skipping"); continue
        bad = age[idx] > MAX_FRAME_AGE_S
        if bad.any():
            # frames missing around some ticks (recorder started late / camera hiccup): keep the longest clean stretch
            good = ~bad
            runs_ = np.split(idx, np.flatnonzero(np.diff(good.astype(int)) != 0) + 1)
            runs_ = [r for r in runs_ if good[np.searchsorted(idx, r[0])]]
            best = max(runs_, key=len) if runs_ else np.zeros(0, int)
            log(f"segment {s}: {bad.sum()} of {len(idx)} ticks have no frame within {MAX_FRAME_AGE_S*1000:.0f} ms "
                f"(max gap {age[idx].max()*1000:.0f} ms); keeping the longest clean stretch of {len(best)} ticks")
            idx = best
            if len(idx) < min_ticks:
                log(f"segment {s}: too short after trimming, skipping"); continue
        plan.append((s, idx))
    if not plan:
        log("nothing to import"); return 0

    ds, n_before = open_dataset(root, repo_id, vcodec)
    n_new = 0
    try:
        for s, idx in plan:
            ep = ds.meta.total_episodes
            task = seg_task.get(s, run["task"]) or run["task"]
            t0 = time.time()
            log(f"episode {ep}: segment {s}, {len(idx)} ticks ({len(idx)/FPS:.1f} s), task {task!r}, "
                f"frame age median {np.median(age[idx])*1000:.1f} / max {age[idx].max()*1000:.1f} ms")
            for k, i in enumerate(idx):
                frame = {"observation.state": state[i].astype(np.float32), "observation.velocity": vel[i].astype(np.float32),
                         "action": action[i].astype(np.float32), "task": task}
                for c in CAMS:
                    frame[f"observation.images.{c}"] = recs[c].frame(int(near[c][0][i]))
                ds.add_frame(frame)
                if k % 300 == 299:
                    log(f"  {k+1}/{len(idx)} frames")
            if dropped_frames(ds):
                raise RuntimeError(f"video encoder dropped {dropped_frames(ds)} frame(s): episode {ep} NOT saved (video would be out of step)")
            ds.save_episode()
            rec = {"episode_index": ep, "run": run_key, "rec": str(rec_dir.resolve()), "segment": int(s), "task": task,
                   "n_frames": int(len(idx)), "ticks": [int(idx[0]), int(idx[-1])],
                   "t_wall": [float(run["t_wall"][idx[0]]), float(run["t_wall"][idx[-1]])],
                   "frame_age_ms": {"median": round(float(np.median(age[idx]) * 1000), 2), "max": round(float(age[idx].max() * 1000), 2)},
                   "ckpt": run["ckpt"], "dry": run["dry"], "joint_offset_vec_deg": np.degrees(run["off14"][:6]).round(3).tolist(),
                   "rec_cap_fps": rec_meta.get("cap_fps"), "source": "policy_rollout", "imported": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "conventions": {"state": "measured joints rad, gripper training raw", "action": "executor commanded target, physical frame, gripper 0=open 1=closed",
                                   "velocity": "finite difference * fps"}, **meta}
            with open(root / "meta" / "rollouts.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_new += 1
            log(f"episode {ep} saved ({time.time() - t0:.0f} s)")
    finally:
        ds.finalize()
    info = json.loads((root / "meta" / "info.json").read_text())
    log(f"done: {n_new} new episode(s); {root.name} now holds {info['total_episodes']} episodes / {info['total_frames']} frames")
    return n_new


def dataset_info(root: Path) -> dict:
    p = root / "meta" / "info.json"
    if not p.exists():
        return {"name": root.name, "episodes": 0, "frames": 0, "exists": False}
    info = json.loads(p.read_text())
    return {"name": root.name, "episodes": info.get("total_episodes", 0), "frames": info.get("total_frames", 0), "exists": True,
            "tasks": info.get("total_tasks", 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="convert", choices=["convert", "info"])
    ap.add_argument("--run", help="executor run directory (holds run.npz)")
    ap.add_argument("--rec", help="WebUI frame recording directory (runs/rec_<ts>)")
    ap.add_argument("--root", required=True, help="dataset directory (created if missing)")
    ap.add_argument("--repo-id", default=None, help="LeRobot repo id stored in info.json (default yam/<root name>)")
    ap.add_argument("--meta", default="{}", help="JSON with extra provenance fields for meta/rollouts.jsonl")
    ap.add_argument("--vcodec", default="h264", help="h264 (training data) | libsvtav1 | h264_nvenc")
    ap.add_argument("--min-ticks", type=int, default=MIN_TICKS)
    ap.add_argument("--force", action="store_true", help="import even if this run+segment is already listed")
    a = ap.parse_args()
    root = Path(a.root)
    if a.cmd == "info":
        print(json.dumps(dataset_info(root)))
        return
    if not a.run or not a.rec:
        ap.error("convert needs --run and --rec")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    n = convert(Path(a.run), Path(a.rec), root, a.repo_id or f"yam/{root.name}", json.loads(a.meta), a.vcodec, a.min_ticks, a.force)
    sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()

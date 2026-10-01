"""Inspect a webui frame recording (runs/rec_<ts>/: <cam>.mjpeg + <cam>.idx.npy + meta.json).
  tools/rec_tools.py list                                  # recordings with duration / frame counts
  tools/rec_tools.py sharp <rec> [--cam left_wrist]        # per-frame sharpness + inter-frame motion over time (find blurry moments)
  tools/rec_tools.py sheet <rec> --t0 12.3 --t1 12.6 [--cam left_wrist] [--every 1]   # contact sheet of frames in [t0, t1] s after start
  tools/rec_tools.py frames <rec> --t0 12.3 --t1 12.6 --cam high --out dir             # dump PNGs
Times are seconds after the recording start (meta t0)."""
import argparse, glob, json, os
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ("high", "left_wrist", "right_wrist")


def load(rec, cam):
    d = rec if os.path.isdir(rec) else os.path.join(ROOT, "runs", rec)
    meta = json.load(open(os.path.join(d, "meta.json"))); idx = np.load(os.path.join(d, f"{cam}.idx.npy"))
    data = np.memmap(os.path.join(d, f"{cam}.mjpeg"), dtype=np.uint8, mode="r")
    return meta, idx, data


def frame(idx, data, i, gray=False):
    t, c, off, n = idx[i]; off, n = int(off), int(n)
    return cv2.imdecode(np.frombuffer(data[off:off + n], np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)


def sharpness(img):
    return float(np.abs(cv2.Sobel(img, cv2.CV_32F, 1, 0)).mean() + np.abs(cv2.Sobel(img, cv2.CV_32F, 0, 1)).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["list", "sharp", "sheet", "frames"]); ap.add_argument("rec", nargs="?")
    ap.add_argument("--cam", default="left_wrist"); ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--out", default=None); ap.add_argument("--window", type=float, default=0.25)
    a = ap.parse_args()
    if a.cmd == "list":
        for d in sorted(glob.glob(os.path.join(ROOT, "runs", "rec_*"))):
            m = json.load(open(os.path.join(d, "meta.json")))
            print(f"{os.path.basename(d)}: {m.get('t1', 0) - m['t0']:.0f} s, {m.get('frames', '?')} frames, session {m.get('tag')}")
        raise SystemExit
    meta, idx, data = load(a.rec, a.cam); ts = idx[:, 0] - meta["t0"]
    sel = np.flatnonzero((ts >= a.t0) & (ts <= a.t1))[:: a.every]
    if a.cmd == "sharp":
        # per window: mean sharpness, mean inter-frame diff (motion), min sharpness frame
        prev = None; rows = []
        for i in sel:
            g = frame(idx, data, i, gray=True); s = sharpness(g); mv = float(np.abs(g.astype(np.int16) - prev).mean()) if prev is not None else 0.0
            rows.append((ts[i], s, mv)); prev = g.astype(np.int16)
        rows = np.array(rows); w = a.window
        print(f"{a.cam}: {len(rows)} frames, {rows[-1,0]-rows[0,0]:.1f} s | window {w} s: t, sharpness (mean/min), motion (mean inter-frame diff)")
        for t in np.arange(rows[0, 0], rows[-1, 0], w):
            m = (rows[:, 0] >= t) & (rows[:, 0] < t + w)
            if m.any():
                print(f"  {t:6.2f}s  sharp {rows[m, 1].mean():5.2f} / {rows[m, 1].min():5.2f}   motion {rows[m, 2].mean():5.2f}")
    elif a.cmd == "sheet":
        sel = sel[:24]
        tiles = [frame(idx, data, i) for i in sel]
        for tl, i in zip(tiles, sel):
            cv2.putText(tl, f"{ts[i]:.3f}s  sharp {sharpness(cv2.cvtColor(tl, cv2.COLOR_BGR2GRAY)):.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cols = 3; rows_ = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows_ * 400, cols * 640, 3), np.uint8)
        for k, tl in enumerate(tiles):
            sheet[(k // cols) * 400:(k // cols + 1) * 400, (k % cols) * 640:(k % cols + 1) * 640] = tl
        out = a.out or os.path.join(ROOT, "runs", os.path.basename(a.rec.rstrip("/")), f"sheet_{a.cam}_{a.t0:.2f}-{a.t1:.2f}.jpg")
        cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]); print("wrote", out, f"({len(tiles)} frames)")
    else:
        os.makedirs(a.out, exist_ok=True)
        for i in sel:
            cv2.imwrite(os.path.join(a.out, f"{a.cam}_{ts[i]:08.3f}.png"), frame(idx, data, i))
        print(f"wrote {len(sel)} frames to {a.out}")

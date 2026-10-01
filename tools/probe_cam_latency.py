"""Camera glass-to-executor latency, measured with the gripper as a physical reference.

Only the LEFT gripper moves (arm joints are held where they are). The gripper is opened, then commanded to
close a little; the wrist camera frames are fetched from the webui as fast as possible (the same path the
executor uses) and the first frame in which the fingers visibly move is found. The latency reported is
  first-visible-motion-in-a-frame  minus  the time the motor position first started to change (from CAN feedback),
so motor response time is excluded; what remains is sensor exposure + USB + driver + ffmpeg + HTTP fetch.
Repeated N times; the gripper ends OPEN. Run with the webui up and no executor running.
  .venv/bin/python tools/probe_cam_latency.py [--n 5] [--side left]
"""
import argparse, sys, time
import cv2, numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from prism_pi.control.executor import RealArm, LEFT_CAN, RIGHT_CAN, ESTOP_FLAG
from prism_pi.rig.hardware import HttpCam

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--side", default="left", choices=["left", "right"])
ap.add_argument("--close", type=float, default=0.35, help="gripper command (0 open .. 1 closed) for the step")
ap.add_argument("--thr", type=float, default=6.0, help="minimum mean abs pixel change to call motion")
a = ap.parse_args()
if ESTOP_FLAG.exists():
    raise SystemExit("e-stop latched")

cam = HttpCam(f"{a.side}_wrist")
arm = RealArm(LEFT_CAN if a.side == "left" else RIGHT_CAN, a.side)
q = arm.state_train()
hold = np.r_[q[:6], 1.0]          # i2rt gripper 1 = open
arm.command(hold); time.sleep(1.0)


def g_norm():
    return float(arm.robot.get_joint_pos()[6])   # 1 open .. 0 closed


def frame_gray():
    f = cam.frame()
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


# finger region: the fingers occupy the bottom of the wrist image; use the bottom 45 %
def finger_band(g):
    return g[220:, :]


res = []
for k in range(a.n):
    arm.command(hold); time.sleep(1.2)
    g0 = g_norm()
    base = np.median([finger_band(frame_gray()) for _ in range(5)], axis=0)
    noise = float(np.mean([np.abs(finger_band(frame_gray()) - base).mean() for _ in range(5)]))
    thr = max(a.thr, 3 * noise)
    t_cmd = time.monotonic()
    arm.command(np.r_[q[:6], 1.0 - a.close])
    t_motor = None; t_seen = None
    while time.monotonic() - t_cmd < 1.5:
        if t_motor is None and abs(g_norm() - g0) > 0.02:
            t_motor = time.monotonic()
        t_f = time.monotonic()
        d = float(np.abs(finger_band(frame_gray()) - base).mean())
        if d > thr:
            t_seen = t_f
            break
    arm.command(hold)
    if t_motor is None or t_seen is None:
        print(f"trial {k}: motor moved {t_motor is not None}, seen {t_seen is not None}, thr {thr:.1f} — skipped")
        continue
    res.append(((t_seen - t_motor) * 1000, (t_seen - t_cmd) * 1000, (t_motor - t_cmd) * 1000))
    print(f"trial {k}: cmd->motor {res[-1][2]:.0f} ms, motor->seen {res[-1][0]:.0f} ms, cmd->seen {res[-1][1]:.0f} ms (thr {thr:.1f})")
    time.sleep(0.5)

arm.command(hold); time.sleep(0.8)
arm.close()
if res:
    r = np.array(res)
    print(f"camera latency (motor motion -> visible in a fetched frame): median {np.median(r[:, 0]):.0f} ms, "
          f"min {r[:, 0].min():.0f}, max {r[:, 0].max():.0f}  (n={len(r)}; fetch loop resolution ~{1000 / 30:.0f} ms)")

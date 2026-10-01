"""Hardware calibration probe: drive one arm so the finger tips sit a set height above the work surface with the
gripper pointing straight down, hold for a ruler measurement, then return to rest and release torque.

IK/FK use the URDF model (prism_pi/control/kinematics.py) in the ARM'S OWN base frame (x = reach direction,
z = up, origin = base mounting plate top / work surface when the plates are flush). During the hold it prints the
model's tip position from the MEASURED joints, so a ruler reading that disagrees = zeroing / kinematic error.

  .venv/bin/python tools/probe_tip_height.py --arm left --plan-only     # offline check
  .venv/bin/python tools/probe_tip_height.py --arm left                 # arm MOVES
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prism_pi import paths  # noqa: E402
from prism_pi.control.kinematics import YamArm, GripperCalibration, solve_ik, top_down_grasp_frame, FINGER_TIP_OFFSET  # noqa: E402

YAM_URDF = str(paths.URDF)
CTRL_HZ = 30.0
REST = np.array([0, 0, 0, -0.08, 0, 0], float)
ESTOP = paths.ESTOP_FLAG
CAN = {"left": "can1", "right": "can0"}


def tip_of(arm: YamArm, q6) -> np.ndarray:
    T = arm.link_poses(np.concatenate([q6, [1.45]]))["gripper"]
    return (T @ np.array([0, 0, -FINGER_TIP_OFFSET, 1.0]))[:3], T[:3, 2]


def interp(q_from, q_to, step):
    n = int(np.ceil(np.abs(q_to - q_from).max() / step)) + 1
    return [q_from + (q_to - q_from) * (i / n) for i in range(1, n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["left", "right"], default="left")
    ap.add_argument("--reach", type=float, default=0.25, help="tip distance ahead of the base centre, m")
    ap.add_argument("--lateral", type=float, default=0.0, help="tip lateral offset in the arm frame, m")
    ap.add_argument("--height", type=float, default=0.01, help="tip height above the work surface, m")
    ap.add_argument("--hover", type=float, default=0.12, help="vertical approach distance above the target, m")
    ap.add_argument("--hold", type=float, default=45.0, help="seconds to hold at the target")
    ap.add_argument("--step", type=float, default=0.02, help="joint-space rad per tick")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    arm = YamArm(YAM_URDF, GripperCalibration(6.351, 1.45))
    # ---- IK: hover, then vertical descent in 1 cm steps; try the finger-opening yaws, then axis-only ----
    zs = list(np.arange(args.height + args.hover, args.height, -0.01)) + [args.height]
    q_way = None
    for mode, yaw in (("full", np.pi / 2), ("full", 0.0), ("full", -np.pi / 2), ("full", np.pi), ("axis", 0.0)):
        q_prev = REST.copy(); cand = []
        for z in zs:
            T = top_down_grasp_frame(np.array([args.reach, args.lateral, z + FINGER_TIP_OFFSET]), yaw=yaw)
            q, err = solve_ik(arm, T, np.concatenate([q_prev, [1.45]]), mode=mode, restarts=12 if not cand else 0)
            tip, zax = tip_of(arm, q)
            if err > 2e-3 or zax[2] < 0.995:
                print(f"  IK {mode} yaw {np.degrees(yaw):.0f}: failed at z={z:.3f} (pos err {err*1000:.1f} mm, axis z {zax[2]:.3f})")
                cand = None; break
            cand.append(q); q_prev = q
        if cand:
            q_way = cand; print(f"IK ok: mode {mode}, opening-axis yaw {np.degrees(yaw):.0f} deg"); break
    if q_way is None:
        raise SystemExit("no IK solution for this target")
    lo, hi = arm.joint_limits[:, 0], arm.joint_limits[:, 1]
    # ---- full joint-space path: rest -> hover -> ... -> target, then reverse ----
    path = []
    for q_from, q_to in zip([REST] + q_way[:-1], q_way):
        path += interp(q_from, q_to, args.step)
    path_down = path
    path_up = path_down[::-1][1:] + interp(q_way[0], REST, args.step)
    tips = np.array([tip_of(arm, q)[0] for q in path_down + path_up])
    print(f"target: tip {args.reach:.3f} m ahead, {args.lateral:+.3f} m lateral, {args.height*100:.1f} cm up; "
          f"gripper axis straight down; {len(path_down)} ticks down ({len(path_down)/CTRL_HZ:.1f} s), {len(path_up)} up")
    print("joint solution at target (rad):", np.round(q_way[-1], 3), "| within URDF limits:",
          bool(np.all(q_way[-1] > lo) and np.all(q_way[-1] < hi)))
    print(f"path check: min tip height {tips[:,2].min()*100:.1f} cm, max reach {tips[:,0].max():.3f} m, "
          f"lateral range {tips[:,1].min():+.3f}..{tips[:,1].max():+.3f} m")
    tf_, zf = tip_of(arm, q_way[-1])
    print(f"model tip at target: ({tf_[0]:.4f}, {tf_[1]:.4f}, {tf_[2]:.4f}) m, gripper -z axis {np.round(-zf, 3)}")
    if args.plan_only:
        return

    if ESTOP.exists():
        raise SystemExit("E-STOP flag latched — clear it in the WebUI first.")
    from prism_pi.control.executor import RealArm
    print(f"== connecting {args.arm} arm ({CAN[args.arm]}) ==")
    robot = RealArm(CAN[args.arm], args.arm)

    def cmd(q6):
        robot.command(np.concatenate([q6, [1.0]]))  # gripper open (i2rt normalized 1 = open)

    def q_meas():
        return robot.state_train()[:6]

    q0 = q_meas()
    print("start joints:", np.round(q0, 3), f"max |dev from rest| {np.abs(q0-REST).max():.2f} rad")
    if np.abs(q0 - REST).max() > 0.7:
        raise SystemExit("ABORT: arm is >0.7 rad from the rest pose; move it near rest by hand first.")
    try:
        def run(seq, label):
            t0 = time.monotonic()
            for i, q in enumerate(seq):
                if ESTOP.exists() or not robot.alive:
                    raise SystemExit(f"E-STOP / chain down during {label}")
                cmd(q)
                dt = t0 + (i + 1) / CTRL_HZ - time.monotonic()
                if dt > 0:
                    time.sleep(dt)
        run(interp(q0, REST, 0.03), "homing")
        time.sleep(0.5)
        run(path_down, "descent")
        print(f"== at target, holding {args.hold:.0f} s — measure the finger tips with a ruler ==")
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.hold:
            if ESTOP.exists() or not robot.alive:
                raise SystemExit("E-STOP / chain down during hold")
            cmd(q_way[-1])
            if int((time.monotonic() - t0) * CTRL_HZ) % 150 == 0:
                qm = q_meas(); tm, zm = tip_of(arm, qm)
                print(f"  t={time.monotonic()-t0:4.1f}s measured joints {np.round(qm,3)} | cmd-meas max {np.abs(q_way[-1]-qm).max():.3f} rad "
                      f"| MODEL tip from measured joints: ahead {tm[0]*100:.1f} cm, lateral {tm[1]*100:+.1f} cm, height {tm[2]*100:.1f} cm; "
                      f"axis tilt {np.degrees(np.arccos(np.clip(zm[2],-1,1))):.1f} deg")
            time.sleep(1.0 / CTRL_HZ)
        print("== returning to rest ==")
        run(path_up, "ascent")
        time.sleep(0.5)
        print("rest joints:", np.round(q_meas(), 3))
    finally:
        robot.close()
        print("torque released")


if __name__ == "__main__":
    main()

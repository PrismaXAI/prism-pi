"""Open-loop replay of a recorded dataset action trajectory on the YAM rig (no model).

Purpose: hardware/convention check. If the arms reach
forward/down to the plate with wrists pointing DOWN, the joint/gripper conventions are
physically right and the remaining rollout problems are visual / initial-state.

Sequence (LIVE mode):  open grippers -> ramp to action[0] -> replay actions[0:N] at 30 Hz
(rate-limited, clamped to the training envelope) -> return to park -> zero torque.
E-stop flag checked every tick.  --dry reads state and sends nothing.

The trajectory file is an .npz you export from your training dataset: for each episode you want to replay, an array
"<name>_action" of shape (T, 14) float, the dataset's ACTION column at 30 Hz in dataset order
[L pan, lift, elbow, wr1, wr2, wr3, grip | R ...] (arm joints rad, gripper 0 = open .. 1 = closed). Pick it with
--traj / --episode <name>.

Usage (web UI running: frames are logged through it):
  .venv/bin/python tools/replay_dataset_actions.py --traj traj.npz --episode ep0 --ckpt <ckpt> --dry         # safe check
  .venv/bin/python tools/replay_dataset_actions.py --traj traj.npz --episode ep0 --ckpt <ckpt> --seconds 10   # ARMS MOVE
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism_pi.rig.hardware import GRIP_CAL, LEFT_CAN, RIGHT_CAN, load_train_stats, STATE_NAMES
from prism_pi.control.executor import RealArm, DryArm, ESTOP_FLAG, HEARTBEAT, CTRL_HZ, MAX_GRIP_STEP

ROOT = Path(__file__).resolve().parent.parent


def split(a14):  # policy gripper 0=open/1=closed -> i2rt 0=closed/1=open
    return (np.concatenate([a14[0:6], [np.clip(1 - a14[6], 0, 1)]]),
            np.concatenate([a14[7:13], [np.clip(1 - a14[13], 0, 1)]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help=".npz with '<episode>_action' arrays (T, 14), see the module docstring")
    ap.add_argument("--episode", required=True, help="name prefix of the action array in --traj")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = half speed)")
    ap.add_argument("--sag-comp", action="store_true",
                    help="add the dataset follower-sag offset (config/sag_table.json) to each commanded arm target")
    ap.add_argument("--hold-at-start", type=float, default=0.0,
                    help="pause this many seconds at the dataset start pose (action[0]) before replaying")
    ap.add_argument("--hold-at-grasp", type=float, default=0.0,
                    help="pause this many seconds at the dataset's first grasp onset (gripper action > 0.3) "
                         "with torque held, so fingertip height can be measured with a ruler")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max-arm-step", type=float, default=0.06)
    ap.add_argument("--max-start-jump", type=float, default=0.30)
    ap.add_argument("--ckpt", required=True, help="checkpoint directory; only used for the action envelope clamp")
    args = ap.parse_args()
    if ESTOP_FLAG.exists():
        raise SystemExit("E-STOP flag latched — clear it in the WebUI first.")

    z = np.load(args.traj)
    acts = z[f"{args.episode}_action"].astype(np.float64)
    n = int(min(len(acts), args.seconds * 30))
    # resample for playback speed
    idx = np.clip(np.arange(0, n, args.speed), 0, n - 1).astype(int)
    acts = acts[idx]
    stats = load_train_stats(args.ckpt)
    a_lo, a_hi = stats["action.min"] - 0.10, stats["action.max"] + 0.10
    a_lo[6] = a_lo[13] = 0.0; a_hi[6] = a_hi[13] = 1.0
    rate = np.array([args.max_arm_step] * 6 + [MAX_GRIP_STEP] + [args.max_arm_step] * 6 + [MAX_GRIP_STEP])

    Arm = DryArm if args.dry else RealArm
    left, right = Arm(LEFT_CAN, "left"), Arm(RIGHT_CAN, "right")
    log = {"cmds": [], "states": []}
    from prism_pi.rig.hardware import HttpCam, CAM_PORTS
    import cv2
    cams = {n: HttpCam(n) for n in CAM_PORTS}
    out = ROOT / "runs" / f"replay_{args.episode}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    def snap(tag):
        for n, c in cams.items():
            try:
                cv2.imwrite(str(out / f"{tag}_{n}.jpg"), cv2.cvtColor(c.frame(), cv2.COLOR_RGB2BGR))
            except Exception as e:
                print("frame save failed:", e)
        print(f"   [frames saved: {tag}]")
    state = lambda: np.concatenate([left.state_train(), right.state_train()])
    try:
        q0 = state()
        arm_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        jump = np.abs(acts[0] - q0)[arm_idx].max()
        print("current state:", np.round(q0, 3))
        print("action[0]:    ", np.round(acts[0], 3))
        print(f"start jump (arm joints): {jump:.3f} rad (limit {args.max_start_jump})")
        if jump > args.max_start_jump:
            raise SystemExit("ABORT: move arms near the dataset rest pose (all arm joints ~0, wr1 ~ -0.08) first.")
        cmd = q0.copy()
        for gi, side in ((6, "left"), (13, "right")):
            _, _, tc, to = GRIP_CAL[side]
            cmd[gi] = np.clip((cmd[gi] - to) / (tc - to), 0, 1)
        park = cmd.copy()

        from prism_pi.control.sag_comp import compensate14
        last_state = [q0.copy()]

        def step_towards(target, r):
            nonlocal cmd
            if ESTOP_FLAG.exists() or not (left.alive and right.alive):
                raise KeyboardInterrupt("E-STOP / chain down")
            if args.sag_comp:
                if len(log["cmds"]) % 10 == 0:
                    last_state[0] = state()
                target = compensate14(np.asarray(target, dtype=float), last_state[0])
            target = np.clip(target, a_lo, a_hi)
            cmd = cmd + np.clip(target - cmd, -r, r)
            l, rr = split(cmd); left.command(l); right.command(rr)
            log["cmds"].append(cmd.copy())
            HEARTBEAT.write_text(json.dumps({"t": time.time(), "mode": "dry" if args.dry else "LIVE",
                                             "task": f"replay {args.episode}", "tick": len(log["cmds"])}))
            time.sleep(1 / CTRL_HZ)

        print("== phase 1: open grippers ==")
        t = cmd.copy(); t[6] = t[13] = 0.0
        for _ in range(45):
            step_towards(t, rate)
        print("   gripper state after open:", np.round(state()[[6, 13]], 3), "(dataset start ~1.34-1.44)")
        print("== phase 2: ramp to action[0] ==")
        for _ in range(45):
            step_towards(acts[0], rate)
        time.sleep(0.3)
        s0 = state(); print("   at dataset start pose: L", np.round(s0[:7], 2), " R", np.round(s0[7:], 2))
        snap("start")
        if args.hold_at_start > 0:
            print(f"== HOLDING {args.hold_at_start:.0f}s at the dataset START pose — compare against the dataset frame 0 ==")
            t_end = time.monotonic() + args.hold_at_start
            while time.monotonic() < t_end:
                step_towards(acts[0], rate)
        print(f"== phase 3: replay {len(acts)} steps at {args.speed}x ==")
        onset = None
        if args.hold_at_grasp > 0:
            g = np.where((acts[:, 6] > 0.3) | (acts[:, 13] > 0.3))[0]
            onset = int(g[0]) if len(g) else None
            print(f"   grasp onset in replayed segment: step {onset} (t={onset/30 if onset is not None else float('nan'):.1f}s)")
        for i, a in enumerate(acts):
            step_towards(a, rate)
            if onset is not None and i == onset:
                s = state()
                print(f"== HOLDING {args.hold_at_grasp:.0f}s at grasp onset — MEASURE fingertip height above the desk ==")
                snap("grasp_onset")
                print("   pose L", np.round(s[:7], 2), " R", np.round(s[7:], 2))
                t_end = time.monotonic() + args.hold_at_grasp
                while time.monotonic() < t_end:
                    step_towards(a, rate)  # keep commanding: feeds the 400 ms motor watchdog
                print("== resuming ==")
            if i % 30 == 0:
                s = state(); log["states"].append(s)
                print(f"t={i/30:5.1f}s  L wr1 {s[3]:+.2f} R wr1 {s[10]:+.2f}  "
                      f"L lift/elbow {s[1]:+.2f}/{s[2]:+.2f}  R {s[8]:+.2f}/{s[9]:+.2f}  grip {s[6]:.2f}/{s[13]:.2f}")
        snap("end")
        print("== returning to park ==")
        park_tgt = compensate14(park, state()) if args.sag_comp else park
        for _ in range(int(20 * CTRL_HZ)):  # bounded: at most 20 s
            if np.abs(park_tgt - cmd).max() <= 0.01:
                break
            step_towards(park, rate * 0.5)
        time.sleep(0.5)
    except KeyboardInterrupt as e:
        print("interrupted:", e)
    finally:
        np.savez(out / "replay.npz", cmds=np.array(log["cmds"]), states=np.array(log["states"]),
                 episode=args.episode, dry=args.dry)
        for a in (left, right):
            try: a.close()
            except Exception: pass
        HEARTBEAT.unlink(missing_ok=True)
        print("torques zeroed; log", out)


if __name__ == "__main__":
    main()

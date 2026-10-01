"""Dry-run pi0.5 inference on the bimanual YAM rig: real cameras, real joint states,
NO actions sent to the arms (motors held at zero torque; arms stay limp).

Pipeline mirrors training exactly:
  - cameras captured at 1920x1200 MJPEG (full 16:10 sensor, same as training rigs),
    downscaled to 640x400 like the training videos
  - observation.state = [left q1..q6, left gripper RAW motor rad, right q1..q6, right gripper RAW]
    (training stats show grippers are raw motor radians, arm joints are joint radians)
  - checkpoint's own pre/post processor pipelines (rename -> normalize -> tokenize)

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python -m prism_pi.rig.dryrun --ckpt <checkpoint dir> \
      --task "<a task string of the checkpoint>" --rounds 3 --cams http
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from prism_pi.paths import ROOT
from prism_pi.policy import relative_actions_compat  # noqa: F401  registers the relative-action step name of our training fork
from prism_pi.rig import mapping as _rig_mapping
from prism_pi.rig.hardware import (
    CAM_PORTS,
    LEFT_CAN,
    RIGHT_CAN,
    STATE_NAMES,
    Cam,
    HttpCam,
    PassiveYamReader,
    grip_to_train,
    load_train_stats,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LeRobot pi0.5 checkpoint directory (or a shim, tools/make_ckpt_shim.py)")
    ap.add_argument("--task", required=True, help="task string, verbatim as in the training data")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "dryrun_out"))
    ap.add_argument("--cams", choices=["v4l2", "http"], default="v4l2",
                    help="http = pull frames from the running web UI instead of opening /dev/video*")
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

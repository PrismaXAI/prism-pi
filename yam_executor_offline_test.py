"""Run yam_executor.main() with NO CAN access: DryArm is replaced by a stub that tracks commands with a
first-order lag (~100 ms, like the sim lag dynamics). Real GPU inference, real WebUI frames. Use it to
check loop timing / chunk alignment / a new checkpoint loading before touching the arms:
  HF_HUB_OFFLINE=1 .venv/bin/python yam_executor_offline_test.py --ckpt <checkpoint dir> --task "..." --dry --duration 12 --out /tmp/x
"""
import time, numpy as np
import yam_executor as ye
from dryrun_inference import GRIP_CAL

class StubArm:
    def __init__(self, channel, side):
        self.side = side
        self.q = np.array([0, 0, 0, -0.08, 0, 0], float)
        self.g = 1.0   # i2rt normalized: 1 = open
        self.alpha = (1/30) / (1/30 + 0.1)
        self.sent = []
        self.t_cmd = []
    def state_train(self):
        _, _, tc, to = GRIP_CAL[self.side]
        return np.concatenate([self.q, [tc + self.g * (to - tc)]]).astype(np.float32)
    alive = True
    def command(self, a7):
        self.q += self.alpha * (a7[:6] - self.q)
        self.g += self.alpha * (a7[6] - self.g)
        self.sent.append(a7.copy()); self.t_cmd.append(time.monotonic())
    def close(self):
        if len(self.t_cmd) > 60:
            dt = np.diff(self.t_cmd[46:]) * 1000   # main loop only (45 gripper-open cmds + the first sync inference)
            print(f"[{self.side}] {len(self.sent)} cmds; tick period ms: mean {dt.mean():.1f} p50 {np.median(dt):.1f} "
                  f"p99 {np.percentile(dt, 99):.1f} max {dt.max():.1f}; ticks > 40 ms: {(dt > 40).sum()}")

ye.DryArm = StubArm
ye.main()

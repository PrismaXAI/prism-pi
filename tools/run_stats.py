"""Summarise a real-rig executor log (runs/<ts>/run.npz): arbitration owner timeline, every gripper hold with
where it started / where the gripper opened, and a rig-side grip-confidence proxy.

A HOLD = gripper state inside the contact band (something between the pads) while the policy commands a close.
It ends either 'released' (the policy opened the gripper) or 'LOST' (the state ran on toward fully closed or the
hold broke while the close command was still on = the fruit slipped / was squeezed out) - the rig analogue of
the simulator's grasp_quality.py lost holds. Closes that never enter the band are 'air' closes.
Positions: pad centre from FK on the MEASURED joints, cm: x lateral (+ right), y ahead of the base line, h height.

usage: .venv/bin/python tools/run_stats.py runs/<ts> [runs/<ts> ...]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism_pi.control.arm_arbiter import ZoneArbiter  # noqa: E402

HZ = 30.0
BAND = (2.4, 4.8)      # gripper state (training-rig rad) with a fruit between the pads (apples 2.5-4.5)
CLOSE_CMD = 0.15


def holds_for(side, o, S, C, A):
    T = len(S); g, gc = S[:, o + 6], C[:, o + 6]
    pad = np.array([A.pad(side, S[i, o:o + 6]) for i in range(T)]) * 100
    closing = gc > CLOSE_CMD
    idx = np.flatnonzero(np.diff(np.r_[0, closing.astype(int), 0]))
    out = []
    for a, b in zip(idx[::2], idx[1::2]):
        b = min(b, T - 1)
        if b - a < 4:
            continue
        seg = g[a:b + 1]
        blocked = (1.34 + gc[a:b + 1] * 5.06 - seg > 0.3) & (seg > 2.25)     # pads stopped well short of the commanded closure = a (large) fruit
        inband = ((seg > BAND[0]) & (seg < BAND[1])) | blocked
        kind = "air" if inband.sum() < 6 else "hold"
        # lost: while the close command is still on, the state runs past the band (pads closing through) after a hold
        lost = False
        if kind == "hold":
            last_in = a + int(np.flatnonzero(inband).max())
            lost = bool(np.any(g[last_in:b + 1] > BAND[1] + 0.2)) or (seg[inband].max() - np.median(seg[inband][:8]) > 0.6)
        r = b
        while r < T - 4 and g[r] > 2.2:
            r += 1
        v = float(np.linalg.norm(pad[min(r + 3, T - 1)] - pad[max(r - 3, 0)]) / (6 / HZ))
        out.append(dict(side=side, t0=a / HZ, t1=b / HZ, kind=kind, lost=lost, cmd=float(np.median(gc[a:b + 1])),
                        state=float(np.median(seg)), state_max=float(seg.max()), grasp=pad[a], maxh=float(pad[a:b + 1, 2].max()),
                        t_open=r / HZ, open=pad[r], speed=v))
    return out, pad


def main():
    A = ZoneArbiter()
    for run in sys.argv[1:]:
        d = np.load(f"{run.rstrip('/')}/run.npz", allow_pickle=True)
        S, C = d["states"], d["cmds"]; n = min(len(S), len(C)); S, C = S[:n], C[:n]
        print(f"=== {run}: {n / HZ:.0f} s, latency {1000 * float(np.mean(d['latency'])):.0f} ms" + (f", grip assist {float(d['grip_assist'])}" if "grip_assist" in d.files else ""))
        if "arb_owner" in d.files and len(d["arb_owner"]):
            own = d["arb_owner"][:n]; ix = np.flatnonzero(np.diff(np.r_[-1, own]))
            print("  owner:", " ".join(f"{i / HZ:.1f}s:{['none', 'L', 'R'][own[i]]}" for i in ix))
        allh = []
        for side, o in (("left", 0), ("right", 7)):
            hs, pad = holds_for(side, o, S, C, A); allh += hs
            print(f"  {side}: max wr1 {S[:, o + 3].max():+.2f}, max reach {pad[:, 1].max():.0f} cm, away from rest {(np.linalg.norm(pad - pad[0], axis=1) > 5).sum() / HZ:.0f} s")
            for h in hs:
                tag = "AIR  " if h["kind"] == "air" else ("LOST " if h["lost"] else "hold ")
                print(f"    {tag}{h['t0']:5.1f}-{h['t1']:5.1f}s cmd {h['cmd']:.2f} state {h['state']:.2f} (max {h['state_max']:.2f}) | grasp ({h['grasp'][0]:+.0f},{h['grasp'][1]:.0f},h{h['grasp'][2]:.0f}) maxh {h['maxh']:.0f}"
                      f" | opened {h['t_open']:.1f}s at ({h['open'][0]:+.0f},{h['open'][1]:.0f},h{h['open'][2]:.0f}) {h['speed']:.0f} cm/s")
        real = [h for h in allh if h["kind"] == "hold"]
        carried = [h for h in real if h["maxh"] > 8 and abs(h["open"][0] - h["grasp"][0]) > 12]
        print(f"  SUMMARY: close attempts {len(allh)}, air {sum(h['kind'] == 'air' for h in allh)}, holds {len(real)}, LOST holds {sum(h['lost'] for h in real)}, carries (lifted >8 cm and moved >12 cm laterally) {len(carried)}")


if __name__ == "__main__":
    main()

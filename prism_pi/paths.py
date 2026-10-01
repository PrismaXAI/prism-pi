"""Where things live. Everything hangs off the repository root, so a checkout is self-contained.

Per-rig configuration is read from config/ (git-ignored files next to tracked *.example.json templates);
run-time state and outputs are written to the repository root (all git-ignored).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# configuration (copy the *.example.json next to each and edit)
CONFIG = ROOT / "config"
RIG_MAPPING = CONFIG / "rig_mapping.json"     # CAN adapters, camera ports, gripper calibration
CHECKPOINTS = CONFIG / "checkpoints.json"     # checkpoints the web UI offers
SAG_TABLE = CONFIG / "sag_table.json"         # optional data-rig sag offsets (leader-action checkpoints)

URDF = ROOT / "assets" / "yam" / "yam.urdf"   # arm kinematics for the executor's FK-based limits

# run-time state shared between the web UI and the executor
ESTOP_FLAG = ROOT / "ESTOP"                   # e-stop latch: present = nothing may command the arms
HEARTBEAT = ROOT / "executor_status.json"
CONTROL = ROOT / "executor_control.json"

# outputs
RUNS = ROOT / "runs"
DATASETS = ROOT / "datasets"
TRT_CACHE = ROOT / "trt_cache"

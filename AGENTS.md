# AGENTS.md: prism-pi

Guide for coding agents (Claude Code and others) setting up and operating this repository for a user.
Read it fully before running anything; every command below is meant to be run from the repository root.

## What this is

The rollout stack we use to run pi0.5 (LeRobot) checkpoints on a bimanual
[I2RT YAM](https://github.com/i2rt-robotics/i2rt) station:

- `prism_pi/policy/server.py`: keeps one checkpoint in VRAM and serves inference on a local socket (K samples per
  observation, real-time chunking, CUDA-graph denoising, optional TensorRT engines). Never touches the robot.
- `prism_pi/control/executor.py`: the 30 Hz control loop. Observes every 25 ticks, executes the action chunk without pausing
  during inference, with rate limits, an action envelope, homing, arm arbitration and an e-stop latch.
- `prism_pi/webui.py`: the operator UI on port 8080 (camera tiles, session start/pause/end, E-STOP, optional
  recording of rollouts as a LeRobot dataset). It owns the cameras and launches the other two.

The companion repository `yamulator` simulates the same station and runs the same control code; use it to
evaluate a checkpoint or an executor change before a live run. This repository does not depend on it.

Modules are run as `python -m prism_pi.<...>` from the repository root (no install step; the root is the
working directory for everything, and run-time state and outputs are written there).

## Safety rules (read first)

This software moves two robot arms. For an agent:

1. **Never start anything that can move the arms unless the user has asked for that specific run in this
   conversation and says they are at the rig with the e-stop within reach.** Earlier approval does not carry over
   to the next run.
2. Commands that move the arms: `-m prism_pi.control.executor` without `--dry`, a UI session without `"dry": true`,
   `tools/identify_arms.py`, `tools/probe_tip_height.py` without `--plan-only`, `tools/probe_cam_latency.py`
   (gripper only), `tools/replay_dataset_actions.py` without `--dry`.
3. Commands that make the arms go limp (zero torque; an arm that is holding a raised pose will drop):
   `-m prism_pi.rig.dryrun`, the executor with `--dry`, the i2rt survey tool, and the E-STOP. Arms must be resting
   or supported first.
4. Safe at any time (no CAN traffic): the policy server and its client, the web UI itself, a UI session with
   `"dry": true`, `-m prism_pi.control.offline_test`, `-m prism_pi.rollout_to_lerobot`, `tools/run_stats.py`,
   `tools/rec_tools.py`, `tools/make_ckpt_shim.py`.
5. The E-STOP (red bar in the UI, or `curl -X POST localhost:8080/estop`) kills the control processes, sends
   motor-off to both buses and writes the latch file `ESTOP`. Do not delete the latch yourself to get a run
   going: the user clears it with "Reset latch" after checking the rig. It is a software stop; the user should
   also know where the arms' power switch is.
6. Do not weaken the guards (start-jump gate, rest-pose check, rate limits, envelope, gripper-state check,
   excursion abort, down-speed cap) to make a run start. A guard that fires is information: report it.
7. The UI listens on all interfaces without authentication. Anyone on the network can start a session. Use it
   on a trusted network only.

## What you need from the user before starting

1. The rig: two YAM arms with linear 4310 grippers, one USB-CAN adapter per arm (socketcan, 1 Mbit/s), three
   UVC cameras (overhead "high", left wrist, right wrist) that deliver 1920x1200 MJPEG; `sudo` for `ip link`.
2. A Linux x86_64 machine with an NVIDIA GPU with 16 GB or more (tested on an RTX 5090 laptop, 24 GB, also with
   the free memory capped at 15.5 GiB to stand in for an RTX 5070 Ti). The server needs about 7 GiB of GPU memory
   in steady state and peaks at about 11 GiB while loading: bf16 weights, the two unused LM heads dropped, the
   PyTorch twins of the TensorRT stages parked in host RAM (`YAM_OFFLOAD_REPLACED=0` keeps them on the GPU, about
   4.5 GiB more). The model is built and cast in host RAM first, so the load needs about 24 GB of free host RAM.
   Loading the model OUTSIDE the server (`--server off`, the simulator's local runner) still puts the fp32 model on
   the GPU, about 16 GB: on a 16 GB card always go through the server.
3. `ffmpeg`, `v4l2` devices under `/dev/v4l/by-path`, `git`, [`uv`](https://docs.astral.sh/uv/).
4. A pi0.5 checkpoint in LeRobot format whose training data follows the conventions in "Checkpoint and rig
   conventions". No checkpoint ships with this repository: ask the user where theirs is and which task strings
   it was trained on.
5. A Hugging Face account that has accepted the licence of the gated `google/paligemma-3b-pt-224` repo
   (only its tokenizer is fetched).

Steps 1-5 of Setup need no robot and are the right first milestone.

## Setup

### 1. Environment

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt -c constraints.txt

About 15 GB installed (torch 2.10.0 with CUDA 12.8, lerobot 0.5.1, transformers 5.3.0, TensorRT 11).
`constraints.txt` pins every package to the validated versions; always pass it. Other LeRobot versions reject the
checkpoint config or change preprocessing. Do not reuse a training environment and do not "upgrade to latest".

### 2. i2rt SDK

    git clone --depth 1 --branch v1.3.5 https://github.com/i2rt-robotics/i2rt i2rt
    git -C i2rt apply ../patches/i2rt-join-control-thread-on-close.patch     # optional: silences a shutdown traceback
    uv pip install --python .venv/bin/python --no-deps -e i2rt

`--no-deps` is deliberate: the SDK's full dependency list does not resolve on Python 3.12; the parts used here
need only what `requirements.txt` already installs. Check:

    HF_HUB_OFFLINE=1 .venv/bin/python -c "import prism_pi.control.executor, prism_pi.policy.client; from i2rt.robots.get_robot import get_yam_robot; print('ok')"

### 3. Tokenizer (gated, one-time download)

The user accepts the licence of `google/paligemma-3b-pt-224` on huggingface.co and provides a token
(`hf auth login`, or `HF_TOKEN=...`); you cannot do that for them. Then:

    .venv/bin/python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224'); print(t('Pick and place various fruits into containers', add_special_tokens=True)['input_ids'][:4])"

It must print a list starting with `2` (`<bos>`), e.g. `[2, 21585, 578, 2040]`. A look-alike tokenizer from an
ungated mirror had the right vocabulary but no `<bos>`; every rollout with it was silently off-distribution
(the policy hedged and closed on air) and it took us days to find. The server and the executor refuse to run
without `[2, ...]`. Afterwards everything runs with `HF_HUB_OFFLINE=1`.

### 4. Checkpoints

Register the checkpoints the UI may run:

    cp config/checkpoints.example.json config/checkpoints.json      # then edit

Each entry: `label`, `path` (the checkpoint's `pretrained_model` directory, absolute or relative to the
repository root), `tasks` (task strings verbatim as in the training data: the prompt is part of the policy input),
`flags` (`"pnp"` = RTC + one-arm-at-a-time arbitration + excursion guard, for pick-and-place; `"bimanual"` = RTC
only, for tasks where both arms work together; or an explicit list of executor arguments).

If loading fails on an unknown `config.json` key (checkpoints saved by a newer or forked LeRobot), make a shim
(filtered `config.json` + symlinks to the weights) and register that instead:

    .venv/bin/python tools/make_ckpt_shim.py /path/to/checkpoints/004000/pretrained_model ckpt/fruits_004000

### 5. Policy server (no robot needed)

    HF_HUB_OFFLINE=1 setsid nohup .venv/bin/python -u -m prism_pi.policy.server --ckpt ckpt/fruits_004000 > policy_server.log 2>&1 < /dev/null &
    .venv/bin/python -m prism_pi.policy.client info        # prints the loaded checkpoint once it is up (about 1 min)

In `policy_server.log` expect `ready: {...}` and `TensorRT vision ON (parity ...)`. Three stages of an inference can
run as TensorRT engines; without any of them everything still works, slower.

| stage | engine | how it gets built | switches |
|---|---|---|---|
| vision (SigLIP on the 3 images) | `trt_vision.py` | automatically on first load (1-3 min, cached in `trt_cache/`) | `YAM_TRT_VISION=0` to disable; `YAM_TRT_PRECISION=bf16` (default) or `fp16` |
| prefix (Gemma, fills the KV cache) | `trt_prefix.py` | one-time export per checkpoint, below (about 1 min, 7 GB of disk) | `YAM_TRT_PREFIX=0`; `YAM_TRT_PREFIX_PRECISION=fp16` (default) or `bf16` |
| action expert (the 10 flow-matching steps) | `trt_expert.py` | one-time export per checkpoint, below (under 1 min, 1.6 GB) | **opt-in**: `YAM_TRT_EXPERT=1`; `YAM_TRT_EXPERT_PRECISION=fp16` (default) or `bf16` |

Exports (stop the server first, they need the VRAM; each precision is its own export; the tools build the model in
host RAM and need about 7 GiB of GPU memory plus the TensorRT builder's workspace, which is capped at what is free):

    HF_HUB_OFFLINE=1 .venv/bin/python -m prism_pi.policy.trt_prefix --ckpt ckpt/fruits_004000 --export --build --check
    HF_HUB_OFFLINE=1 .venv/bin/python -m prism_pi.policy.trt_expert --ckpt ckpt/fruits_004000 --export --build --check

`--check` prints the engine's deviation from the PyTorch path and counts non-finite outputs (must be 0). The
switches are environment variables of the server process: when the UI starts the server it inherits the UI's
environment, so set them where `rig_up.sh` / the UI is launched, e.g. `YAM_TRT_EXPERT=1 ./rig_up.sh`.

At load the server compares each engine's actions with the PyTorch path on the same seed and keeps the engine only
if they are finite and well inside the seed-to-seed spread; the log says `TensorRT <stage> ON (parity ...)` or `OFF`.
At run time a request whose actions are not finite fails instead of returning them (the executor parks on a failed
inference) and the expert engine is dropped for the following requests.

Server-side latency per inference on the RTX 5090 laptop, RTC on (median of 300, logged rig observations):

| configuration | latency |
|---|---|
| PyTorch only (bf16, CUDA-graph denoising) | about 110 ms |
| vision bf16 + prefix fp16 engines (the default once the prefix is exported) | 80 ms |
| + expert bf16 engine | 74 ms |
| + expert fp16 engine | 66 ms |
| vision, prefix and expert all fp16 (`YAM_TRT_PRECISION=fp16 YAM_TRT_EXPERT=1`) | 63 ms |

Against a pure fp32 PyTorch reference on 120 logged observations (same noise seed), every configuration above stays
within 0.004-0.007 rad (median of the per-chunk maximum over arm joints; two different noise seeds differ by 0.29 rad),
the expert engines included, and none produced a NaN or Inf, also on 100 off-distribution inputs. The expert engine
In closed loop (`yamulator` through this server, 30 paired scenes per configuration, RTC + arbitration) the lift
counts were 22 of 30 without the expert engine and 23 to 26 with it (fp16, bf16, all-fp16): no difference beyond the
run-to-run noise of that sample size. The expert engine is opt-in because it is the newest stage and has not run on
the rig yet: turn it on after `--check` passes for your checkpoint, and judge it by rollouts (simulator, then rig).

Stop the server with `.venv/bin/python -m prism_pi.policy.client shutdown`.
One model at a time: while the server is up, loading the model in another process can run out of VRAM.
`client info` reports `gpu_used_gib` (whole device, as nvidia-smi sees it) next to `vram_gib` (torch tensors only;
the TensorRT engines live outside that count). Engines that are not cached yet are built BEFORE the model is moved
to the GPU, so a first start on a new GPU takes a few minutes longer but needs no more memory than a warm start.

### 6. Rig mapping (needs the hardware, no motion)

    cp config/rig_mapping.example.json config/rig_mapping.json      # then fill in, see below

- **CAN adapters.** `ip -br link | grep can` lists them; `udevadm info -a /sys/class/net/can0 | grep -m1 serial`
  gives each serial. The code convention is fixed: **LEFT arm = `can1`, RIGHT arm = `can0`**, where "left" is the
  arm at the bottom-left of the overhead image. Put `serial -> can0/can1` in `can_by_serial`; `./rig_up.sh`
  renames and brings the interfaces up at 1 Mbit/s accordingly (the i2rt guide
  `i2rt/docs/guides/set-persistent-can-ids.md` covers persistent udev names).
- **Cameras.** `ls /dev/v4l/by-path/ | grep video-index0`. Enter each camera's path up to the USB port (without
  the `:1.0-video-index0` suffix) under `cameras`. Identical cameras share a serial, so they are addressed by
  port: re-plugging into another port changes the mapping. One camera per USB link at 120 fps; behind a shared
  hub set `"capture_fps": 60`.
- Which arm is which: with the user present and their go-ahead, `.venv/bin/python tools/identify_arms.py --channel can1`
  wiggles only the wrist roll of the arm on that bus by +-0.3 rad. It must be the LEFT arm. Which camera is
  which: cover a lens and watch the tiles in the UI.
- The cameras must deliver 1920x1200 MJPEG (the whole 16:10 sensor, scaled to 640x400 like the training
  frames, no crop). Another sensor format means editing the ffmpeg arguments in `CamThread.run` of
  `prism_pi/webui.py`; keep the full field of view the training camera had. Leave the cameras on auto exposure.

### 7. Gripper calibration (needs the hardware, no commanded motion)

The gripper state fed to the policy is the raw motor angle mapped onto the training data's convention, per
gripper. Measure yours, with the user moving the fingers by hand:

    .venv/bin/python tools/calibrate_gripper.py --side left
    .venv/bin/python tools/calibrate_gripper.py --side right

and paste the two printed entries into `"grip_cal"` in `config/rig_mapping.json`:
`[closed_raw, open_raw, train_closed, train_open]`. The last two numbers are the gripper STATE values of the
training data at closed and at typical open (ours: 6.351 / 1.45 left, 6.392 / 1.34 right); take them from your
dataset. Without `grip_cal` the code uses our rig's values, which are wrong for any other gripper; the executor
then aborts at start-up ("fits 0 calibrated ranges" or "gripper state after opening"). This tool is new in the
public release and has not been run on hardware yet: sanity-check its output (about 5 rad of travel).

### 8. Start the station

    ./rig_up.sh

Renames and brings up the CAN adapters (sudo), reports a latched e-stop without clearing it, starts the web UI
on port 8080 and prints a status line. It moves no arms and is safe to run twice. Open `http://<machine-ip>:8080`.
Expect: header `IDLE`, `can0:up can1:up`, three live camera tiles. The UI starts the policy server itself at the
first session and restarts it when another checkpoint is selected (about 1 min).

## Verification ladder

Go one rung at a time; stop and report at the first one that fails.

1. **Pipeline without the robot** (works with no CAN adapter attached; `config/rig_mapping.json` must exist, but its
   camera entries may point at absent devices: the tiles show OFFLINE and the policy gets placeholder frames):

       curl -s -X POST localhost:8080/session/start -H 'content-type: application/json' \
            -d '{"ckpt": "fruits", "task": "<task string>", "mode": "timed", "duration": 30, "dry": true}'
       curl -s localhost:8080/status

   This runs the real executor against stub arms (`prism_pi/control/offline_test.py`) with the real cameras and
   policy server. In `runs/ui_<stamp>_timed.log` expect `first-action jump`, `== running: sim-matched scheme`,
   `== done: 900 ticks, ...` with an install offset of 3-8 ticks, and tick periods of 33 ms. Output goes to
   `/tmp/yam_ui_test/`.
2. **Passive state read** (arms limp, user present): `HF_HUB_OFFLINE=1 .venv/bin/python -m prism_pi.rig.dryrun --ckpt <ckpt> --task "<task>" --cams http`
   with the UI running and the policy server stopped (it loads the model itself). Per joint it prints state,
   first action and the training range. With the arms resting at the dataset start pose, the first action must
   be within about 0.25 rad of the state and nothing flagged `OUT_OF_TRAIN_RANGE`; the gripper states must
   read like the training data's open value. If not, the joint or gripper conventions do not match: stop.
3. **Executor dry** (arms limp, commands computed but not sent; UI and policy server running):
   `HF_HUB_OFFLINE=1 .venv/bin/python -m prism_pi.control.executor --ckpt <ckpt> --task "<task>" --dry --duration 12 --server on`.
   Nothing opens the grippers in dry mode, so the user opens them by hand first (or pass `--keep-grippers`);
   otherwise the gripper-state check aborts the run, which is that check working.
4. **First live run**, only on the user's explicit go-ahead: from the UI, timed, 60 s, user's hand on the
   e-stop, arms at rest (roughly upright and folded, `[0, 0, 0, -0.08, 0, 0]`), grippers free to open. The
   executor homes to the rest pose, opens the grippers, checks the first action, then runs.

## Operating

- **Sessions** (UI): pick checkpoint and task, TIMED (30-150 s) or CONTINUOUS. PAUSE holds both arms in place
  under power with the policy stopped (reset the scene, optionally change the task), RESUME continues from a
  fresh observation, END parks the arms and releases torque. Toggles: grip assist, frame sync, smooth commands,
  record frames, LeRobot dataset.
- **Command line** equivalent of the pick-and-place preset (UI and policy server running):
  `HF_HUB_OFFLINE=1 .venv/bin/python -m prism_pi.control.executor --ckpt <ckpt> --task "<task>" --duration 60 --rtc --arbitrate --server on --abort-wr1 0.6`.
  `-m prism_pi.control.executor -h` documents every option; defaults are the validated ones.
- **Logs**: `runs/ui_<stamp>_<mode>.log` (console of the session), `runs/ui_<stamp>/run.npz` (every tick: state,
  command, chunks, latency, arbitration) and `frames/` (the exact images given to the policy),
  `policy_server.log`, `webui.log`. `.venv/bin/python tools/run_stats.py runs/ui_<stamp>` summarises holds, releases and
  arbitration; `tools/rec_tools.py` inspects frame recordings.
- **Recording rollouts as training data**: the "LeRobot dataset" toggle records the session's frames at the
  capture rate and converts them with the per-tick log into `datasets/<name>/` (LeRobot v3, 30 fps, one episode
  per PAUSE/RESUME segment; schema and conventions in the docstring of `prism_pi/rollout_to_lerobot.py`). Success labels
  are not recorded.
- **After an e-stop** the arms are limp and the latch is set; the next session start re-enables the motors after
  the user resets the latch. If the motors stop answering after an e-stop while the links show UP, the CAN
  adapter is wedged: `sudo ip link set canX down` then `up type can bitrate 1000000` (or `./rig_up.sh`).
- **After a power cycle of an arm** the gripper motor can re-reference by a whole turn; the executor detects
  it (`limit shift +1 turn(s)`) and aborts if the gripper state after opening is implausible.
- **Restart the UI**: `kill $(ss -ltnp | grep ':8080' | grep -o 'pid=[0-9]*' | cut -d= -f2)`, then `./rig_up.sh`.
  Environment overrides: `YAM_UI_PORT`, `YAM_CAP_FPS`, `YAM_POLICY_SOCK`, `YAM_UI_TEST=1` (every session dry).
- **Read-only bus check** (arm limp while it runs, Ctrl-C to stop):
  `.venv/bin/python i2rt/i2rt/motor_drivers/dm_driver.py --channel can0 --print-pos`.

## Checkpoint and rig conventions the code assumes

The stack is written for one data convention (ours). A checkpoint trained on other conventions needs the matching
constants changed, not just another `--ckpt`.

| item | convention | where it lives |
|---|---|---|
| policy | LeRobot `pi05`, chunk of 50 actions at 30 Hz, 10 flow steps | `CTRL_HZ`, `CHUNK` in `control/executor.py`; `policy/server.py` |
| images | `observation.images.{high,left_wrist,right_wrist}`, 640x400 RGB from the full 16:10 sensor | `IMG_KEYS` in `policy/server.py`; the checkpoint's own rename step maps them to the model's names |
| state (14) | `[L pan, lift, elbow, wr1, wr2, wr3, gripper, R ...]`: arm joints in rad as i2rt reports them, gripper in the data rig's raw motor rad (closed about 6.35, open about 1.4) | `RealArm.state_train` in `control/executor.py`, `GRIP_CAL` in `rig/hardware.py` / `grip_cal` in the rig mapping |
| action (14) | same order; arm joints absolute rad, or deltas from the chunk-start state when `use_relative_actions` is set in the checkpoint's `config.json` (grippers stay absolute); gripper 0 = open .. 1 = closed | `split()` in `control/executor.py`, `policy/relative_actions_compat.py` |
| start of an episode | both arms at `[0, 0, 0, -0.08, 0, 0]`, grippers open (state 1.0-2.3) | `REST_POSE`, homing and gripper check in `control/executor.py` |
| action envelope | the checkpoint's normalizer stats +-0.10 rad (state stats for relative-action checkpoints) | `control/executor.py` |
| leader vs follower actions | if the data's actions are leader-arm commands and the follower sagged below them, `--sag-comp auto` adds that offset when `config/sag_table.json` exists (`config/sag_table.example.json` = ours); never for checkpoints marked `FOLLOWER_ACTIONS` or with "follower" in `train_config.json` | `control/sag_comp.py` |
| excursion guard | parks the arms if a wrist_1 exceeds `--abort-wr1` rad (0 disables): a pick-and-place signature, wrong for tasks that legitimately pitch the wrists up | `control/executor.py` |

Rig geometry matters as much as software. What we measured, in the simulator and on the rig:

- Arm base plates level with the work surface (2 cm of height difference costs success, 5 cm breaks it).
- The overhead camera must frame the scene like the training frames (object rows and columns in the 640x400
  image); a lens wider than the training lens hurts, a narrower one does not. Tilt is the sensitive axis.
- Wrist cameras on the same mount as the data rig, to a few millimetres.
- The joint STATE convention must match the data rig's: a constant offset of a few degrees on shoulder lift made
  an absolute-action policy close above the fruit (`--joint-offset-vec-deg 0 -4 0 0 0 0` was the remedy);
  relative-action checkpoints are far less sensitive.
- Large diffuse lighting and auto exposure near the training frames' brightness. Scene colours matter little.
- Objects the training scenes always contain (for us: a plate, two containers) should be present and fixed down.

## Repository map

| path | purpose |
|---|---|
| `prism_pi/webui.py` | operator UI, camera capture (one ffmpeg per camera), e-stop, session control, frame recorder |
| `prism_pi/control/` | `executor.py` (the control loop; `offline_test.py` = the same loop on stub arms) and the command shaping it applies: `arm_arbiter.py` (one arm at a time in the shared zone), `tool_offset.py` (downward-speed cap, pre-close advance), `kinematics.py` (URDF forward kinematics / IK those two need), `sag_comp.py` |
| `prism_pi/policy/` | `server.py` and `client.py` (`info`, `load <ckpt>`, `shutdown`), server-side acceleration (`fast_denoise.py` CUDA graph, `trt_vision.py` / `trt_prefix.py` / `trt_expert.py` TensorRT engines in `trt_cache/`), `relative_actions_compat.py` |
| `prism_pi/rig/` | `mapping.py` (reads `config/rig_mapping.json`), `hardware.py` (zero-torque arm reader, camera sources, CAN names, `GRIP_CAL`), `dryrun.py` (passive inference check) |
| `prism_pi/rollout_to_lerobot.py` | rollout + frame recording -> LeRobot dataset |
| `prism_pi/paths.py` | every file location (config, e-stop latch, state files, outputs) in one place |
| `config/` | `rig_mapping.example.json`, `checkpoints.example.json`, `sag_table.example.json`; your copies without `.example` live next to them |
| `tools/` | `identify_arms.py`, `calibrate_gripper.py`, `probe_tip_height.py` (kinematic zero check with a ruler), `probe_cam_latency.py`, `replay_dataset_actions.py` (open-loop convention check), `make_ckpt_shim.py`, `run_stats.py`, `rec_tools.py` |
| `assets/yam/yam.urdf` | arm kinematics for the FK-based limits |
| `rig_up.sh`, `patches/` | station start-up; optional patch for the i2rt SDK |

Counterparts in `yamulator`: the simulator runs the same control and client code as `yamulator/control/`
(`arm_arbiter.py`, `tool_offset.py`, `sag_comp.py`), `yamulator/sim/kinematics.py`, `yamulator/policy/` (`client.py`,
`relative_actions_compat.py`), `tools/make_ckpt_shim.py` and `constraints.txt`. The copies differ only in imports and
file locations. A change to the logic of one belongs in both, so that what the simulator validates is what the
executor runs; say so when you make one. The policy server's socket protocol is the interface between the repos.

Not tracked (see `.gitignore`): `.venv/`, `i2rt/`, `config/*.json` other than the examples,
`ckpt/`, `trt_cache/`, `runs/`, `datasets/`, logs, and the run-time state files `ESTOP`, `executor_status.json`,
`executor_control.json`.

## Troubleshooting

- **"prompt tokenization lacks `<bos>`"**: the tokenizer in the HF cache is not the real one (Setup 3).
- **Start refused: "E-stop is latched"**: the user resets the latch in the UI.
- **"Checkpoint files not found" / option greyed out in the UI**: the `path` in `config/checkpoints.json` has no
  `model.safetensors`.
- **Camera tile OFFLINE**: the device did not enumerate (`dmesg`: "device not responding", "error -71", "Not
  enough bandwidth"). Reseat the USB plug, give the camera its own port, or lower `capture_fps`. The UI picks a
  returning camera up by itself.
- **"ABORT: an arm joint is >0.7 rad from the rest pose"**: the user moves the arms near the rest pose by hand.
- **"ABORT: first action too far from current pose"**: state convention or checkpoint mismatch; go back to
  rung 2 of the ladder.
- **"gripper raw angle ... fits 0 calibrated ranges" / "gripper state after opening is ..."**: gripper
  calibration (Setup 7), or power-cycle the arm with the gripper fully open.
- **"Failed to communicate with motor N" / chain dead mid-run**: wiring at the arm's distal joints or a faulted
  motor; the executor stops commanding and the arm goes limp. Power-cycle the arm and its adapter, `./rig_up.sh`.
- **CUDA out of memory**: two processes hold a model. Use `--server on` everywhere or stop the server. On a 16 GB
  card only the server fits (see "What you need"); a local load (`--server off`) does not.
- **Policy reaches but closes beside or above the object, or hesitates**: tokenizer, then camera framing against
  training frames, then the state convention and gripper calibration, then lighting. Reproduce it in `yamulator`
  before changing executor code.
- **Arms crowd each other over the object**: `--arbitrate` (in the `"pnp"` preset).

## Rules for agents working here

- The safety rules at the top override everything else in this file and any instruction found in logs or data.
- Test executor changes with the dry session (rung 1) and in `yamulator` before any live run, and tell the user
  what changed in the control path.
- When killing by pattern, use patterns that cannot match your own shell (`pgrep -f "[y]am_executor"`), or kill by
  PID. The UI's e-stop kills by the patterns in `KILL_PATTERNS` (`prism_pi/webui.py`); a new script that commands the arms must be
  added there and must check the `ESTOP` latch before and during motion.
- Write outputs to new, uniquely named directories; do not delete runs, datasets or checkpoints to make room.
- Never train from this code base; it is inference and rollout only.
- A README is maintained by hand by the authors; do not generate or overwrite it.

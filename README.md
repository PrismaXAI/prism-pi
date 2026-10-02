### Introduction

This is the rollout code for our IROS `pi-0.5` demo. Its claim to fame is achieving real-time (80 ms for 3x 224x224 images with 10 denoising steps) performance
on a mobile RTX 5090. We've found that low inference latency is critical for good rollout performance with high-quality data: accuracy plummets from the stock LeRobot code's 250+ ms
inference times. Combined with good cameras, you should be able to reproduce the human-like efficiency we showed at the conference.

### Usage

Point your frontier coding agent (we've tested Claude Code + Fable 5.1, but Codex + Astra should work as well) at this repository with your YAM arms and cameras connected to your
deployment machine and tell it to set up the rollout code. In practice, this requires a few prompts:

- First, make sure you can read out all three of your cameras at max frame rate simultaneously. This usually requires connecting two cameras to USB-A ports and one camera to a USB-C port -- happily, any 5090 laptop will have full-speed Thunderbolt 5 ports which sit on a separate root hub from the USB-A ports, so an external USB card is not necessary.
- Ask your agent to clone this repo and set it up, making sure the Web UI runs and the timings for the forward pass are sane. The dtypes for each stage of the inference pipeline (SigLIP, prefix, expert) are configurable -- we've tested bf16 SigLIP + fp16 prefix + bf16 expert, but fp16 expert is 2x faster and should work fine.
- To run inference you will need TensorRT exports for your checkpoint. Your agent can do this for you; the scripts are in the repo under `prism-pi/policy`.
- Identify the arms -- your agent can do this by moving the wrists and watching for changes -- and have it write you a JSON rig mapping file following the example in `config/`.
- Open the Web UI on a remote machine or your phone and carefully run some real inference. Make sure there is a robust network connection between the frontend and the backend since the E-stop
goes through the connection as well -- operating from a remote machine protects you from the arms, but not the arms from themselves.

By default, the execution server uses an arbiter to prevent both arms from reaching for the same pick and place target -- we found this was helpful for minimizing the risk of camera damage
during tests. Obviously, you want this disabled for bimanual tasks. 

To get the best performance, you will need fast cameras. Cheap cameras often have some level of internal buffering giving them over 1 frame of latency -- if you are using UVC cameras we
recommend getting one that supports 90+ Hz operation. Real machine vision cameras (RealSense, ZED) should have 1 frame of latency. Unfortunately the codebase does not support RealSense
out of the box but it should be easy to add (and ZED will never be supported since there isn't a reasonable computer with 3x GMSL2 interfaces and a good GPU). We use the ELP AR0234 USB 3.0
cameras from Amazon. Attached to 2 m USB 3.0 extenders these seem to give OK performance for demo-like runtimes, are very cheap, and can be replaced on-site in two days if something breaks.

### Internals

`prism-pi` achieves low latencies without pruning or distillation by simply recovering performance left on the table by other inference codes. Stock openpi inference code uses a naive eager
Pytorch implementation which is 2-3x slower than an optimized implementation. This repo uses a fast TensorRT engine for the Gemma prefix which yields a 2x improvement over `torch.compile` and a 4x speedup over eager. Additionally, we discovered that an eager implementation of the expert is pathologically slow, launching thousands of small individual kernels and causing the GPU to spend a lot of time doing very little -- compiling the expert yielded a drastic improvement in performance.

Finally, a GeForce-specific quirk is that GeForce-branded GPUs don't really support bf16 -- it is capped to 1/2x peak performance -- to prevent you from using them for training. Swapping the dtypes to fp16 for this compute-limited task provides a nearly 2x improvement in performance over the default bf16 with no loss in accuracy.

### About Us

prism-pi is brought to you by [PrismaX](http://prismax.ai). We provide end-to-end services for data collection and model rollout on YAM and other popular embodiments, with over 6000 hours a month of capacity across the most challenging dexterous tasks in the industry, operators with thousands of hours of experience, and sensor, end effector, and scenario customization to meet you and your researchers' needs.

# Third-party material in this repository

| path | source | licence |
|---|---|---|
| `assets/yam/yam.urdf` (YAM arm + linear 4310 gripper kinematics) | i2rt SDK, https://github.com/i2rt-robotics/i2rt | MIT, Copyright (c) I2RT Robotics |
| `patches/i2rt-join-control-thread-on-close.patch` (a change to the i2rt SDK's `dm_driver.py`) | derived from the i2rt SDK | MIT, Copyright (c) I2RT Robotics |

Installed at setup time, not redistributed here: the i2rt SDK (cloned into `i2rt/`), LeRobot, PyTorch, TensorRT
and the other packages in `requirements.txt`; the `google/paligemma-3b-pt-224` tokenizer (gated, Hugging Face).

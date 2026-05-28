# Run a Policy on a Robot

> **Preview:** `PolicyRuntime` and the CLI are planned APIs. The examples below document the target design.

## Python API

```python
from physicalai.runtime import PolicyRuntime, SyncExecution
from physicalai.inference import InferenceModel
from physicalai.robot import SO101
from physicalai.capture import UVCCamera

runtime = PolicyRuntime(
    fps=30,
    robot=SO101(port="/dev/ttyACM0"),
    model=InferenceModel.load("./exports/act_policy"),
    cameras={
        "wrist": UVCCamera(device="/dev/video0", width=640, height=480),
    },
    execution=SyncExecution(mode="chunk"),
)

runtime.run(duration_s=60)
```

## From Config

Write a runtime configuration file.

```yaml
# runtime.yaml
runtime:
  class_path: physicalai.runtime.PolicyRuntime
  init_args:
    fps: 30
    robot:
      class_path: physicalai.robot.so101.SO101
      init_args:
        port: /dev/ttyACM0
    model:
      class_path: physicalai.inference.InferenceModel
      init_args:
        export_dir: ./exports/act_policy
    cameras:
      wrist:
        class_path: physicalai.capture.UVCCamera
        init_args:
          device: /dev/video0
          width: 640
          height: 480
    execution:
      class_path: physicalai.runtime.SyncExecution
      init_args:
        mode: chunk
```

Load and run from Python.

```python
from physicalai.runtime import PolicyRuntime

runtime = PolicyRuntime.from_config("runtime.yaml")
runtime.run(duration_s=60)
```

Or run from the CLI.

```bash
physicalai run --config runtime.yaml --duration-s 60
```

## Component Responsibilities

| Object           | Owns                  |
| ---------------- | --------------------- |
| `InferenceModel` | policy inference      |
| `PolicyRuntime`  | robot loop and timing |
| `Execution`      | where inference runs  |
| `Robot`          | hardware IO           |
| `Camera`         | image capture         |

# Run a Policy

> **Preview:** `PolicyRuntime` and the CLI are planned APIs. The examples below document the target design.

Use `PolicyRuntime` to run a trained policy on real hardware. The runtime handles the control loop: reading cameras, building observations, running inference, and sending actions to the robot.

```python
from physicalai.inference import InferenceModel
from physicalai.runtime import PolicyRuntime, SyncExecution
from physicalai.robot import SO101
from physicalai.capture import UVCCamera

model = InferenceModel.load("./exports/act_policy")
robot = SO101(port="/dev/ttyACM0")
cameras = {
    "wrist": UVCCamera(device="/dev/video0", width=640, height=480),
}

runtime = PolicyRuntime(
    fps=30,
    robot=robot,
    model=model,
    cameras=cameras,
    execution=SyncExecution(mode="chunk"),
)

runtime.run(duration_s=60)
```

The equivalent CLI command uses the same runtime configuration.

```bash
physicalai run --config runtime.yaml --duration-s 60
```

The minimal runtime configuration looks like this.

```yaml
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

At a high level, the runtime loop follows this sequence.

```text
read robot observation
read camera frames
build observation dict
run inference
pop one action from chunk
send action to robot
sleep until next tick
```

# Run a Policy

Use `RobotRuntime` with a `PolicySource` action source to run a trained policy on real hardware. The runtime handles the control loop — reading cameras, sending actions — while the action source builds observations and runs inference.

```python
from physicalai.inference import InferenceModel
from physicalai.runtime import RobotRuntime, PolicySource, SyncExecution
from physicalai.robot import SO101
from physicalai.capture import UVCCamera

model = InferenceModel("./exports/act_policy")
robot = SO101(port="/dev/ttyACM0")
cameras = {
    "wrist": UVCCamera(device="/dev/video0", width=640, height=480),
}

runtime = RobotRuntime(
    fps=30,
    robot=robot,
    action_source=PolicySource(model=model, execution=SyncExecution()),
    cameras=cameras,
)

with runtime:
    runtime.run(duration_s=60)
```

The equivalent CLI command uses the same runtime configuration.

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

The minimal runtime configuration looks like this.

```yaml
runtime:
  robot:
    class_path: physicalai.robot.SO101
    init_args:
      port: /dev/ttyACM0
      calibration: ./calibration.json
  action_source:
    class_path: physicalai.runtime.PolicySource
    init_args:
      model:
        class_path: physicalai.inference.InferenceModel
        init_args:
          export_dir: ./exports/act_policy
      execution:
        class_path: physicalai.runtime.SyncExecution
  cameras:
    wrist:
      class_path: physicalai.capture.UVCCamera
      init_args:
        device: /dev/video0
        width: 640
        height: 480
  fps: 30
```

At a high level, the runtime loop follows this sequence.

```text
read robot observation
read camera frames
ask the action source for the next action
send action to robot
sleep until next tick
```

# Write Runtime Config

A runtime config describes a robot control workflow before execution starts.

```yaml
# runtime.yaml
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
        device: /dev/v4l/by-id/usb-Example_Wrist_Camera-video-index0
        width: 640
        height: 480
  fps: 30
```

Run the same config from the CLI:

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

Nested components use the same `class_path` and `init_args` shape.

```yaml
class_path: module.ClassName
init_args:
  key: value
```

The config file remains passive data. The workflow starts only when `RobotRuntime.run()` is called.

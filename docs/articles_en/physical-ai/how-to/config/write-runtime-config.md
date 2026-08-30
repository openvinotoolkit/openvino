# Write Runtime Config

A runtime config describes a robot control workflow before execution starts.

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
          device: /dev/v4l/by-id/usb-Example_Wrist_Camera-video-index0
          width: 640
          height: 480
    execution:
      class_path: physicalai.runtime.SyncExecution
      init_args:
        mode: chunk
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

The config file remains passive data. The workflow starts only when `PolicyRuntime.run()` is called.

# Configuration

> **Preview:** The config system (`physicalai.config`, `PolicyRuntime.from_config`, and the CLI) are planned APIs. The content below documents the target design. Currently, `ComponentSpec` lives in `physicalai.inference.manifest`.

The config system is intended to make Python, YAML, CLI, and Studio payloads use the same workflow shape.

```python
runtime = PolicyRuntime.from_config("runtime.yaml")
runtime.run()
```

```bash
physicalai run --config runtime.yaml
```

## Layers

```text
Config
  typed constructor args for one class

ComponentSpec
  target + args for one instantiable component

Workflow config
  user-authored workflow before execution

Manifest
  exported package metadata after build/export

Orchestrator
  live object that executes the workflow
```

## ComponentSpec

Direct class mode:

```yaml
class_path: physicalai.capture.UVCCamera
init_args:
  device: /dev/v4l/by-id/usb-Example_Camera-video-index0
  width: 640
  height: 480
```

> **Tip:** Use stable device paths (`/dev/v4l/by-id/...`) in config files. Index-based paths like `/dev/video0` can change after reboot.

Registry mode:

```yaml
type: uvc
device: /dev/v4l/by-id/usb-Example_Camera-video-index0
width: 640
height: 480
```

If both `class_path` and `type` are present, `class_path` takes precedence.

## Typed Config

Typed configs are useful when constructor validation and IDE support matter.

```python
@dataclass
class Pi05Config(Config):
    chunk_size: int = 50
    n_action_steps: int = 50

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size")
```

Typed configs do not decide which class to instantiate. They only validate and carry constructor arguments.

```python
cfg = Pi05Config(chunk_size=50)
policy = instantiate_obj(cfg, target_cls=Pi05)
```

## Execution Boundary

Configuration objects remain passive data. Orchestrators are responsible for creating live objects and executing workflows.

```python
config = RuntimeConfig.load("runtime.yaml")
runtime = PolicyRuntime.from_config(config)
runtime.run()
```

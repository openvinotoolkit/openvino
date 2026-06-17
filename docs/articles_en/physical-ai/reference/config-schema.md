# Config Schema Reference

> **Preview:** The config system and `PolicyRuntime` are planned APIs. The schemas below document the target design.

Config files use `class_path` and `init_args` to describe explicit component construction.

## ComponentSpec

Direct class mode:

```yaml
class_path: package.module.ClassName
init_args:
  key: value
```

Registry mode:

```yaml
type: registered_name
key: value
```

The `ComponentSpec` fields are listed below.

| Field        | Type   | Description                             |
| ------------ | ------ | --------------------------------------- |
| `class_path` | string | Fully qualified import path             |
| `init_args`  | object | Constructor keyword arguments           |
| `type`       | string | Registered short name                   |
| extra fields | any    | Flat constructor args for registry mode |

The core rules are straightforward.

- A component spec must include either `class_path` or `type`.
- If both fields are present, `class_path` takes precedence.
- Nested component specs are instantiated recursively.

## RuntimeConfig

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
    execution:
      class_path: physicalai.runtime.SyncExecution
      init_args:
        mode: chunk
```

The most common runtime fields are listed below.

| Field                         | Type            | Description                |
| ----------------------------- | --------------- | -------------------------- |
| `runtime`                     | `ComponentSpec` | Runtime orchestrator       |
| `runtime.init_args.fps`       | number          | Control loop frequency     |
| `runtime.init_args.robot`     | `ComponentSpec` | Robot implementation       |
| `runtime.init_args.model`     | `ComponentSpec` | Inference model            |
| `runtime.init_args.execution` | `ComponentSpec` | Execution strategy         |
| `runtime.init_args.cameras`   | mapping         | Optional camera components |
| `runtime.init_args.callbacks` | list            | Optional runtime callbacks |

## InferenceConfig

```yaml
model:
  class_path: physicalai.inference.InferenceModel
  init_args:
    export_dir: ./exports/act_policy
    backend: openvino
    device: CPU
```

The most common inference fields are listed below.

| Field                        | Type            | Description                |
| ---------------------------- | --------------- | -------------------------- |
| `model`                      | `ComponentSpec` | Inference model component  |
| `model.init_args.export_dir` | string          | Exported package directory |
| `model.init_args.backend`    | string          | Backend name or `auto`     |
| `model.init_args.device`     | string          | Backend device or `auto`   |

## Config vs Manifest

| Schema          | Use                                                      |
| --------------- | -------------------------------------------------------- |
| Workflow config | A workflow config describes a workflow before execution. |
| Manifest        | A manifest describes an exported package after export.   |

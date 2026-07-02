# Manifest Schema Reference

The on-disk manifest is `manifest.json`. YAML is shown here for readability.

## Top Level

```yaml
format: policy_package
version: "1.0"
policy: {}
model: {}
hardware: {}
```

Fields:

| Field      | Type   | Description                               |
| ---------- | ------ | ----------------------------------------- |
| `format`   | string | Manifest format, usually `policy_package` |
| `version`  | string | Manifest schema version                   |
| `policy`   | object | Policy identity and source metadata       |
| `model`    | object | Artifacts and inference pipeline          |
| `hardware` | object | Expected robot and camera metadata        |

## Policy

```yaml
policy:
  name: pi05
  source:
    repo_id: physical-ai/example
    class_path: physicalai.policies.pi05.Pi05
```

## Model

```yaml
model:
  n_obs_steps: 1
  artifacts:
    openvino: model.xml
  runner:
    type: action_chunking
    chunk_size: 50
  preprocessors:
    - type: normalize
      artifact: stats.safetensors
  postprocessors:
    - type: denormalize
      artifact: stats.safetensors
```

Fields:

| Field            | Type                  | Description                        |
| ---------------- | --------------------- | ---------------------------------- |
| `n_obs_steps`    | integer               | Observation history length         |
| `artifacts`      | mapping               | Logical artifact name to file path |
| `runner`         | `ComponentSpec`       | Inference runner                   |
| `preprocessors`  | list[`ComponentSpec`] | Preprocessing pipeline             |
| `postprocessors` | list[`ComponentSpec`] | Postprocessing pipeline            |

## Hardware

```yaml
hardware:
  robots:
    - name: main
      type: SO101
      state:
        shape: [6]
        dtype: float32
        order: [shoulder, elbow, wrist, gripper, lift, rotate]
      action:
        shape: [6]
        dtype: float32
  cameras:
    - name: wrist
      shape: [3, 224, 224]
      dtype: uint8
```

Robot and camera specs are metadata only. The runtime config still selects the concrete hardware classes.

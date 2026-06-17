# CLI: Infer

> **Preview:** The CLI is a planned API. The examples below document the target design.

Run inference from a config or exported policy package.

```bash
physicalai infer --config inference.yaml
```

Example config:

```yaml
model:
  class_path: physicalai.inference.InferenceModel
  init_args:
    export_dir: ./exports/act_policy
    backend: openvino
    device: CPU
```

The Python equivalent:

```python
model = InferenceModel.load("./exports/act_policy")
action = model.select_action(observation)
```

Use `physicalai run` for robot control loops. Use `physicalai infer` for offline inference or testing outside a runtime loop.

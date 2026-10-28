# Write Inference Config

Use an inference config when you need to author an inference pipeline outside an exported manifest.

```yaml
model:
  class_path: physicalai.inference.InferenceModel
  init_args:
    export_dir: ./exports/act_policy
    backend: openvino
    device: CPU
```

If the manifest already contains the required runner, artifacts, processors, and hardware metadata, prefer loading from the manifest instead.

```python
model = InferenceModel("./exports/act_policy")
action = model.select_action(observation)
```

Use workflow config to express user-authored intent. Use a manifest to describe exported package metadata.

To export a model's supplied constructor arguments from Python, call
`Config.from_instance(model)`. Rebuild that trusted local recipe with
`config.instantiate()`; construction loads the model package
and can allocate substantial resources.

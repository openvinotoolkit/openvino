# Configure Preprocessing and Postprocessing

Preprocessors run before model execution. Postprocessors run after model execution. Together, they define the input and output adaptation around the runner.

The manifest can declare both stages.

```yaml
model:
  runner:
    type: action_chunking
    chunk_size: 50
  artifacts:
    openvino: model.xml
  preprocessors:
    - type: normalize
      artifact: stats.safetensors
  postprocessors:
    - type: denormalize
      artifact: stats.safetensors
```

The same components can also be declared with explicit class paths.

```yaml
preprocessors:
  - class_path: physicalai.inference.preprocessors.StatsNormalizer
    init_args:
      artifact: stats.safetensors
```

Pipeline shape:

```text
observation
  -> preprocessors
  -> runner
  -> postprocessors
  -> action output
```

Use `type` for registered built-in components. Use `class_path` when you want an explicit import path.

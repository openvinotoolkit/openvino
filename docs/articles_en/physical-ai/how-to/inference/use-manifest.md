# Use a Manifest

An exported policy package contains a `manifest.json` file.

```text
exports/act_policy/
├── manifest.json
├── model.xml
└── stats.safetensors
```

In most cases, you load the package through `InferenceModel`.

```python
model = InferenceModel("./exports/act_policy")
```

If you need to inspect metadata directly, load the manifest itself.

```python
from physicalai.inference.manifest import Manifest

manifest = Manifest.load("./exports/act_policy/manifest.json")
print(manifest.model.runner)
print(manifest.model.artifacts)
```

## Declare inference callbacks

Callbacks use the same `ComponentSpec` syntax and declaration order as
preprocessors and postprocessors. Built-in callbacks can use their registered
short name with flat constructor arguments, or a fully qualified class path
with `init_args`:

```yaml
model:
  callbacks:
    - type: rldx1_vtc
      video_length: 4
      video_stride: 2
    - class_path: physicalai.inference.callbacks.LatencyMonitor
      init_args:
        window_size: 100
```

The RLDX-1 callback above creates temporal image windows before preprocessing
without Python-side callback wiring. Manifest callbacks receive `on_load`,
`on_predict_start`, `on_predict_end`, and `on_reset` lifecycle events.

When constructing `InferenceModel`, an explicitly supplied `callbacks` list
takes precedence over the manifest, including an explicitly supplied empty
list. Invalid, unknown, or incompatible callback specifications fail while the
model is loading.

Use manifests to describe exported artifacts. Use workflow config to author training, inference, or runtime workflows before execution.

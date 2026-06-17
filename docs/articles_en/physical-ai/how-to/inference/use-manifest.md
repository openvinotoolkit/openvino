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
model = InferenceModel.load("./exports/act_policy")
```

If you need to inspect metadata directly, load the manifest itself.

```python
from physicalai.inference.manifest import Manifest

manifest = Manifest.load("./exports/act_policy/manifest.json")
print(manifest.model.runner)
print(manifest.model.artifacts)
```

Use manifests to describe exported artifacts. Use workflow config to author training, inference, or runtime workflows before execution.

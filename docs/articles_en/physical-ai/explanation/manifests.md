# Manifests

A manifest describes an exported policy package. It tells the runtime what artifacts exist and how the inference pipeline should be reconstructed.

```text
export/
├── manifest.json
├── model.xml
└── stats.safetensors
```

## Example

```yaml
format: policy_package
version: "1.0"

policy:
  name: pi05
  source:
    class_path: physicalai.policies.pi05.Pi05

model:
  artifacts:
    openvino: model.xml
  runner:
    type: action_chunking
    chunk_size: 50
  preprocessors:
    - type: normalize
      artifact: stats.safetensors

hardware:
  robots:
    - name: main
      type: SO101
  cameras:
    - name: wrist
      type: uvc
    - name: overhead
      type: realsense
```

## Manifest vs Workflow Config

| Data            | Meaning                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| Workflow config | A workflow config describes the desired workflow before running or exporting. |
| Manifest        | A manifest describes the concrete exported package after build or export.     |

To load the package, use the exported directory.

```python
model = InferenceModel.load("./export")
```

To inspect the metadata directly, load the manifest itself.

```python
manifest = Manifest.load("./export/manifest.json")
```

The two schemas can share `ComponentSpec`, but they still serve different purposes.

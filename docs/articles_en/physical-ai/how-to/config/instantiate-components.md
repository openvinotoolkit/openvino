# Instantiate Components

A `Config` recipe points at one class and the arguments for its constructor.

```yaml
class_path: physicalai.capture.UVCCamera
init_args:
  device: /dev/video0
  width: 640
  height: 480
```

## Build from a recipe

`Config.instantiate()` creates a new object from a trusted local recipe. It runs
`__init__` only—you still call `connect()`, `run()`, or similar yourself.

```python
from physicalai.config import Config

config = Config.from_dict({
    "class_path": "physicalai.capture.UVCCamera",
    "init_args": {"device": "/dev/video0", "width": 640, "height": 480},
})
camera = config.instantiate()
camera.connect()
```

## Save how an object was created

Add `@export_config` to a class so Physical AI can record the constructor
arguments you actually passed (omitted defaults stay omitted).

```python
from physicalai.capture import UVCCamera
from physicalai.config import Config

camera = UVCCamera(device="/dev/video0", width=640, height=480)
config = Config.from_instance(camera)
```

Nested components use the same recipe shape inside `init_args`. Values must be
JSON-friendly: paths become strings, tuples become lists, and invalid floats are
rejected.

```python
Config.from_instance(camera).save("camera.yaml")
```

## Safety

`class_path` loads and runs Python code on your machine. Use your own config
files or other sources you trust. Skip instantiation for metadata or messages
that came from another process or the network.

Inference **manifests** (policy exports) use a separate format with aliases and
artifact paths. Use `Config` for runtime construction YAML; use manifest APIs
when loading an exported policy package.

# Cameras

Cameras expose a small capture interface for connecting to a device and retrieving frames.

```python
camera.connect()
frame = camera.read_latest()
camera.disconnect()
```

## Read Modes

| Method          | Behavior                      | Use                                 |
| --------------- | ----------------------------- | ----------------------------------- |
| `read()`        | next frame, blocking          | recording or complete frame streams |
| `read_latest()` | newest frame, non-blocking    | real-time control                   |
| `async_read()`  | async wrapper around `read()` | async applications                  |

## Runtime Use

Control loops usually care more about freshness than completeness.

```python
observation["image.wrist"] = wrist_camera.read_latest()
```

Camera instances are not thread-safe. Use one thread per camera instance or add external synchronization.

## Device Identity

`discover()` reports each camera's `hardware_payload`: backend-specific
identifying fields (a serial, a sensor port, a bus path, ...) with no
guaranteed key and no guaranteed stability across reboot or replug — shape
varies by backend and device. To reopen a specific camera, hand its
`hardware_payload` (or another identity string, e.g. a serial) back to the
backend as `device_id`; it resolves to the best-matching device and raises if
more than one device matches equally well. Persist `hardware_payload` only if
you know your device reports something durable (e.g. a real USB serial).

## SharedCamera

`SharedCamera` lets one publisher subprocess own a camera's exclusive hardware
connection while any number of subscribers read frames over iceoryx2. It
satisfies the same `Camera` protocol as a direct driver.

Construction uses `camera=` or `from_config()`, matching `SharedRobot`. Prefer
`from_config()` (or YAML) when sharing. A disconnected `@export_config` camera
can be passed directly to `camera=`; this exports its recipe rather than
handing an open device into the child. The publisher always opens fresh. Never
keep a direct camera connected to the same device while sharing: another
connected holder causes open failure.

```python
from physicalai.capture import SharedCamera
from physicalai.config import Config

# Prefer config-only / disconnected export — no live direct camera held open
shared = SharedCamera.from_config(
    {
        "class_path": "physicalai.capture.UVCCamera",
        "init_args": {"device": "/dev/video0", "backend": "v4l2"},
    },
)
# Equivalent from a disconnected live driver:
# shared = SharedCamera.from_config(Config.from_instance(driver))
shared.connect()

# Safe to read from multiple threads/processes
frame = shared.read_latest()
```

`create_camera(..., shared=True)` remains a convenience for shareable built-ins
(`uvc`, `realsense`, `basler`, `ip`) that packs a type into `SharedCamera(camera=...)`.

`from_config()` derives `service_name` as
`physicalai/camera/<ClassName>/<device_id>/frame` for any `class_path`,
including third-party drivers. Pass `service_name=` explicitly if two camera
classes share a class name and a device id.

`SharedCamera` is the recommended approach for production deployments where multiple consumers need camera frames. It avoids the need for manual synchronization and handles frame distribution efficiently.

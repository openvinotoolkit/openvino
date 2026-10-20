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

## SharedCamera

For multi-process or multi-thread access, use `SharedCamera`. It wraps any camera and handles IPC transport automatically.

```python
from physicalai.capture import SharedCamera, UVCCamera

shared = SharedCamera(UVCCamera(device="/dev/video0"))
shared.connect()

# Safe to read from multiple threads/processes
frame = shared.read_latest()
```

`SharedCamera` is the recommended approach for production deployments where multiple consumers need camera frames. It avoids the need for manual synchronization and handles frame distribution efficiently.

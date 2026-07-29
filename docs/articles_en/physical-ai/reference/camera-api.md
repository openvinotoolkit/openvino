# Camera API Reference

Camera implementations are expected to inherit from `Camera`.

```python
class Camera:
    def connect(self, timeout: float = 5.0) -> None: ...
    def disconnect(self) -> None: ...
    def read(self, timeout: float = 2.0) -> Frame: ...
    def read_latest(self) -> Frame: ...
    async def async_read(self, timeout: float = 2.0) -> Frame: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def device_id(self) -> str: ...
```

## Color Mode

```python
class ColorMode(StrEnum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
```

## Camera Type

```python
class CameraType(StrEnum):
    UVC = "uvc"
    REALSENSE = "realsense"
    GENICAM = "genicam"
    BASLER = "basler"
```

> **Note:** IP camera support may be available through the UVC backend but is not verified or tested.

## Read Semantics

| Method          | Semantics                                                      |
| --------------- | -------------------------------------------------------------- |
| `read()`        | Returns the next frame, blocks, and preserves sequence.        |
| `read_latest()` | Returns the newest frame, does not block, and may skip frames. |
| `async_read()`  | Provides an async wrapper around `read()`.                     |

Camera instances are not thread-safe.

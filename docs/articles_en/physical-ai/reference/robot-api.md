# Robot API Reference

Robot implementations are expected to satisfy the `Robot` protocol.

```python
class Robot(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> RobotObservation: ...
    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None: ...
    def is_connected(self) -> bool: ...

    @property
    def joint_names(self) -> list[str]: ...
```

## `RobotObservation`

```python
class RobotObservation(Protocol):
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None
    images: dict[str, Frame] | None
```

## Requirements

- The `joint_positions` order must match `joint_names`.
- The action passed to `send_action()` must match `joint_names`.
- `disconnect()` must leave the hardware in a safe stationary state.
- `connect()` should be idempotent or fail with a clear error.

# Robots

Robots implement a structural interface, so inheritance is not required.

```python
class MyRobot:
    joint_names = ["shoulder", "elbow", "wrist"]

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> MyObservation: ...
    def send_action(self, action, *, goal_time: float = 0.1) -> None: ...
    def is_connected(self) -> bool: ...
```

## Observation

A robot observation should expose at least the following fields.

```python
joint_positions: np.ndarray
timestamp: float
sensor_data: dict[str, np.ndarray] | None
images: dict[str, Frame] | None
```

## Action Contract

`send_action()` receives a single action vector whose order matches `joint_names`.

```python
action.shape == (len(robot.joint_names),)
```

Robot implementations are responsible for leaving the hardware in a safe state on disconnect.

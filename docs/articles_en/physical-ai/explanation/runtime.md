# Runtime

`RobotRuntime` runs the control loop on robot hardware. It owns hardware I/O, the callback lifecycle, and timing — while a required, pluggable `action_source` owns the actual decision of what action to send each tick. `PolicySource` wraps a trained policy (model + execution strategy + action queue); `TeleopSource` reads a leader arm; custom logic can implement the `ActionSource` protocol directly.

```python
runtime = RobotRuntime(
    fps=30,
    robot=robot,
    action_source=PolicySource(model=model, execution=SyncExecution()),
)

with runtime:
    runtime.run(duration_s=60)
```

## Responsibilities

| Component        | Owns                                                             | Does not own                |
| ---------------- | ---------------------------------------------------------------- | --------------------------- |
| `InferenceModel` | model load, preprocess, inference, postprocess                   | robot loop timing           |
| `Execution`      | where and when inference runs                                    | robot IO                    |
| `ActionQueue`    | action chunks and buffering                                      | model inference             |
| `PolicySource`   | wiring model + execution + action queue into one action per tick | robot IO, loop timing       |
| `RobotRuntime`   | observe, call the action source, send action, callbacks, timing  | policy math, decision logic |
| `Robot`          | hardware connection, observations, actions                       | policy inference            |

## Loop

The runtime loop follows this general pattern:

```text
while running:
    if stop_requested() or duration_reached():
        break
    robot_state, camera_frames = read_observation()
    action = action_source.update(robot_state, camera_frames, step)
    action = on_action_ready(action)  # callback hook, may transform
    send_action_to_robot(action)
    on_action_sent(action)            # callback hook, notification only
    sleep_until_next_tick()
```

The exact observation structure and merging strategy may change as the API stabilizes. Everything left of `action_source.update()` — deciding whether to run inference, pulling from the queue, holding the last action — is internal to the action source; `RobotRuntime` itself only ever sees one action per tick.

## Stopping

A stop takes effect between ticks, never inside one. The tick already underway finishes and sends its action, so the robot is never left halfway through a command. That means a stop is not instant: it lands once the current tick is done, and a tick that happens to be waiting on inference or a slow robot read takes as long as it takes.

Use `runtime.stop()` when the code asking for the stop lives in the same program. Use `run(stop_event=...)` when it does not: pass any object with an `is_set()` method, and one process can stop a session running in another. The runtime checks both, so whichever comes first ends the run.

Stopping is not the same as shutting down. It ends the control loop, but the robot, cameras, and callbacks stay active, so one runtime can stop and run again. Each run emits its own `start` and `shutdown` lifecycle events. `last_run_reason` and the `shutdown` event's `reason` field say why a run ended.

`disconnect()` is final. It closes callbacks and disconnects hardware exactly once; create a new runtime to connect again. A failed `connect()` already rolls back partial hardware setup, so retry it by calling `connect()` again without calling `disconnect()`. This keeps callback resources such as files and background threads usable across runs while giving them one clear disposal point.

## Execution Modes

> **Preview:** `RemoteExecution` is a planned API.

| Mode               | Where inference runs | Use                              |
| ------------------ | -------------------- | -------------------------------- |
| `SyncExecution()`  | runtime thread       | simple deployments and debugging |
| `AsyncExecution()` | worker thread        | avoid blocking the control loop  |
| `RemoteExecution`  | remote server        | planned API                      |

## Product Workflows

HIL, recording, highlight, and DAgger should be composed through callbacks until they justify reusable runtime primitives.

```python
class HILCallback:
    def on_action_ready(self, *, action, step):
        if teleop.enabled:
            return teleop.read_action()
        return action
```

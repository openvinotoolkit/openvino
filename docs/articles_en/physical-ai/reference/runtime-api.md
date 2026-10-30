# Runtime API Reference

## `RobotRuntime`

`RobotRuntime` is the main orchestrator for running a policy (or teleop, or any custom control logic) on hardware. It always takes an explicit, pluggable `action_source`.

```python
RobotRuntime(
    robot: Robot,
    action_source: ActionSource,
    fps: float,
    cameras: Mapping[str, Camera] | None = None,
    callbacks: Sequence[RuntimeCallback] = (),
)
```

The most important methods are shown below.

```python
runtime.connect() -> None
runtime.disconnect() -> None
runtime.run(*, duration_s: float | None = None, stop_event: StopSignal | None = None) -> int
runtime.stop() -> None
runtime.last_run_reason -> RunReason | None
```

`RobotRuntime` also supports context-manager usage so connections are cleaned up automatically. A "step" is one iteration of the control loop at `fps`: read an observation, get one action from `action_source`, and send it to the robot. `run()` returns the number of steps completed this run — there is no aggregate stats object. Other stats are read directly off the objects the caller already holds, e.g. `runtime.action_source.action_queue.total_pops` or `execution.inference_count`. Each `run()` starts those counters from zero, so they describe the latest run rather than a running total.

Callbacks belong to the runtime. They remain open across consecutive `run()` calls and receive a separate `start` and `shutdown` lifecycle event for each one. `disconnect()` means the caller is finished with the runtime: it closes callbacks and hardware exactly once, and the runtime cannot reconnect. If `connect()` fails, it rolls back partially connected hardware; call `connect()` again directly to retry.

```python
with RobotRuntime(...) as runtime:
    steps = runtime.run(duration_s=60)
```

## Stopping a Run

`stop()` is thread-safe and may be called before, during, or between runs. `run(stop_event=...)` takes any object with `is_set()`, so `threading.Event` and `multiprocessing.Event` both work. Both are polled at the top of each tick — whichever trips first ends the run, and the tick in flight still completes and sends its action.

```python
from physicalai.runtime import StopSignal   # Protocol: is_set() -> bool
```

Calling `stop()` when no run is active is remembered rather than ignored: the next `run()` sees it, returns immediately, and reports zero steps. The request is cleared once a run ends, so one runtime can serve consecutive sessions.

Running again starts a fresh session rather than continuing the old one. Actions still queued from the previous run are dropped, because they were computed from observations that are now out of date. If a background worker is still inside the model when the next run starts, `run()` waits a moment for it and then raises `RuntimeError` if it is still busy, rather than letting two runs share one model.

`run()` returns the step count. `last_run_reason` reports why it ended, and the same string is emitted as `reason` in the `shutdown` lifecycle event.

| `last_run_reason`  | Cause                                                         |
| ------------------ | ------------------------------------------------------------- |
| `stop_requested`   | `runtime.stop()` was called, or `stop_event.is_set()` is true |
| `duration_elapsed` | `duration_s` was reached                                      |
| `interrupted`      | `KeyboardInterrupt` (swallowed; `run()` returns normally)     |
| `error`            | Any other exception, which then propagates to the caller      |

`stop_event` is absent from the config schema and the CLI, since a live synchronisation object has no serialisable form; `physicalai run` stops via `--run.duration_s` or Ctrl+C.

## `ActionSource`

`ActionSource` is the protocol a developer implements to plug custom decision logic into `RobotRuntime`. Three required methods, nothing optional — no capability protocols, no `isinstance` checks anywhere in the runtime.

```python
class ActionSource(Protocol):
    def connect(self, *, bus: _CallbackBus, session_id: str) -> None: ...
    def update(self, robot_state: RobotObservation, camera_frames: Mapping[str, Frame], step: int) -> np.ndarray: ...
    def disconnect(self) -> None: ...
```

`bus` is an internal, runtime-scoped callback bus injected by `RobotRuntime` on every `run()`; action sources typically only forward it into an `Execution` strategy (see `PolicySource` below) rather than using it directly.

The action-source implementations shipped today are listed below.

| Class          | Purpose                                                                |
| -------------- | ---------------------------------------------------------------------- |
| `PolicySource` | runs a trained model through an `Execution` strategy + `ActionQueue`   |
| `TeleopSource` | reads a leader arm and forwards its observation as the follower action |

```python
PolicySource(
    model: InferenceModel,
    execution: Execution,
    action_queue: ActionQueue | None = None,
    *,
    task: str | None = None,
)

source.reset(*, reset_model: bool = True) -> None
source.warmup(observation: dict[str, Any]) -> None

TeleopSource(
    leader: Robot,
    *,
    to_action: Callable[[RobotObservation], np.ndarray] | None = None,
)
```

`reset()` starts a fresh policy episode without stopping the execution worker. It discards queued, pending, and in-flight actions, clears the last action, and resets model state by default. If inference is already running, reset waits for that model call to finish before resetting model state, while rejecting its stale result. Model or execution reset errors propagate to the caller. Pass `reset_model=False` only when model state should continue across the episode boundary. Custom execution strategies must implement `Execution.reset()` before they can support this operation.

`warmup()` accepts model-ready input, such as `source.to_model_input(robot_state, camera_frames)`, and seeds the queue before the next control tick. It is optional: `update()` performs the same warmup lazily after `connect()` or `reset()`.

## `Execution`

```python
class Execution:
    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None: ...
    def maybe_request(self, observation: dict[str, Any]) -> None: ...
    def warmup(self, sample_observation: dict[str, Any]) -> None: ...
    def reset(self, *, reset_model: bool = True) -> None: ...
    def stop(self) -> None: ...
    @property
    def chunk_size(self) -> int: ...
```

The execution implementations shipped today are listed below.

| Class            | Purpose                               |
| ---------------- | ------------------------------------- |
| `SyncExecution`  | runs inference in the runtime thread  |
| `AsyncExecution` | runs inference in a background thread |

> **Preview:** `RemoteExecution` is a planned API and is not part of the current package release.

## `ActionQueue`

```python
queue.push_chunk(chunk)
action = queue.pop()
queue.clear()
```

The action queue owns runtime buffering, merging, smoothing, and the policy for handling late results.

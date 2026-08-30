# Runtime API Reference

## `PolicyRuntime`

`PolicyRuntime` is the main orchestrator for running a policy on hardware.

```python
PolicyRuntime(
    robot: Robot,
    model: InferenceModel,
    execution: Execution,
    fps: float,
    cameras: Mapping[str, Camera] | None = None,
    action_queue: ActionQueue | None = None,
    callbacks: Sequence[RuntimeCallback] = (),
)
```

The most important methods are shown below.

```python
runtime.connect() -> None
runtime.disconnect() -> None
runtime.run(duration_s: float | None = None) -> RunStats
```

`PolicyRuntime` also supports context-manager usage so connections are cleaned up automatically.

```python
with PolicyRuntime(...) as runtime:
    stats = runtime.run(duration_s=60)
```

## `Execution`

```python
class Execution:
    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None: ...
    def maybe_request(self, observation: dict[str, np.ndarray]) -> None: ...
    def warmup(self, sample_observation: dict[str, np.ndarray]) -> None: ...
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

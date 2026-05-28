# Runtime API Reference

> **Preview:** `PolicyRuntime` and related APIs are planned. The signatures below document the target design.

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
    callbacks: Sequence[Callback] = (),
    return_to_home: bool = False,
)
```

The most important methods are shown below.

```python
runtime.run(duration_s: float | None = None) -> None
runtime.stop() -> None
runtime.close() -> None
```

You can also construct the runtime from a config file.

```python
runtime = PolicyRuntime.from_config("runtime.yaml")
```

## `Execution`

```python
class Execution:
    def start(self, action_queue: ActionQueue, model: InferenceModel) -> None: ...
    def maybe_request(self, observation: Mapping[str, Any]) -> None: ...
    def warmup(self, sample_observation: Mapping[str, Any], n: int = 2) -> None: ...
    def stop(self) -> None: ...
```

The expected execution implementations are listed below.

| Class             | Purpose                                      |
| ----------------- | -------------------------------------------- |
| `SyncExecution`   | runs inference in the runtime thread         |
| `AsyncExecution`  | runs inference in a thread or process worker |
| `RemoteExecution` | requests inference from a remote server      |

## `ActionQueue`

```python
queue.push_chunk(chunk)
action = queue.pop_or_none()
queue.clear()
```

The action queue owns runtime buffering, merging, smoothing, and the policy for handling late results.

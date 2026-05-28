# Inference

`InferenceModel` is the runtime API for exported policies. It loads the exported package, runs preprocessing and postprocessing, and produces actions.

```python
model = InferenceModel.load("./exports/act_policy")
action = model.select_action(observation)
```

## Pipeline

```text
observation
  -> preprocessors
  -> runner
  -> postprocessors
  -> action
```

## APIs

| Method                              | Use                                   |
| ----------------------------------- | ------------------------------------- |
| `select_action(observation)`        | Returns one action immediately.       |
| `predict_action_chunk(observation)` | Returns a chunk for runtime queueing. |
| `reset()`                           | Clears state for a new episode.       |
| `close()`                           | Releases backend resources.           |

## Chunked Policies

Chunk-producing policies still support `select_action()`. The caller does not need to branch on runner type.

```python
if cursor.empty():
    cursor.push_chunk(model.predict_action_chunk(obs))

return cursor.pop()
```

The cursor is a convenience inside the model layer. It is not the runtime action queue and it should not be treated as one.

## Runtime Boundary

Use `select_action()` for scripts, tests, demos, and evaluation loops.

Use `predict_action_chunk()` through `PolicyRuntime` when the policy is driving a robot.

```text
PolicyRuntime
  -> Execution.maybe_request(obs)
  -> InferenceModel.predict_action_chunk(obs)
  -> ActionQueue.push_chunk(chunk)
  -> ActionQueue.pop_or_none()
  -> robot.send_action(action)
```

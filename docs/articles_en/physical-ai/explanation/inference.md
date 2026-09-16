# Inference

`InferenceModel` is the runtime API for exported policies. It loads the exported package, runs preprocessing and postprocessing, and produces actions.

```python
model = InferenceModel("./exports/act_policy")
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

## Performance Evaluation

`InferenceLatencyBenchmark` measures per-chunk latency of an `InferenceModel` outside the runtime loop. It is intended for backend, device, and export-configuration comparisons; it is not a robot-loop profiler.

```python
from physicalai.benchmark.performance.inference_benchmark import InferenceLatencyBenchmark
from physicalai.inference import InferenceModel

model = InferenceModel("./exports/act_policy", device="CPU")
metrics = InferenceLatencyBenchmark(max_iters=500, warmup_iters=5).run(model)
```

The run consists of a warmup phase followed by a measured loop bounded by both an iteration cap and a wall-clock budget, whichever is reached first.

```text
warmup_iters  -> measured loop -> metrics
                 (stop on max_iters OR max_duration OR inputs exhausted)
```

When no `inputs` iterable is provided, the benchmark generates random inputs according to `model.input_features` specifications,
so it can run without a recorded dataset. Pass a custom iterable to benchmark against real observations.

The reported metrics (`num_iters`, `min_iter_time`, `max_iter_time`, `mean_iter_time`, `median_iter_time`, `std_iter_time`, `avg_warmup_iter_time`) are per-iteration seconds and reflect the full preprocess → runner → postprocess pipeline.

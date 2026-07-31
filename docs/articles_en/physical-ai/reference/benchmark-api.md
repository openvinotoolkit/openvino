# Benchmark API Reference

## `InferenceLatencyBenchmark`

```python
InferenceLatencyBenchmark(
    max_iters: int | None = 1000,
    warmup_iters: int = 1,
    max_duration: int | None = 60000,
)
```

Measures per-chunk latency of an `InferenceModel`. The measured loop stops at whichever bound is reached first: `max_iters`, `max_duration` (milliseconds), or input exhaustion. Pass `None` to disable a bound.

### `run`

```python
metrics = benchmark.run(model, inputs=None)
```

`inputs` is an iterable of observation dicts compatible with `model`. When `None`, random inputs are generated from `model.input_features`
specifications; this requires the exported package to declare input features.

Runs `warmup_iters` warmup iterations followed by the measured loop and returns a dict of per-iteration seconds:

| Key                    | Meaning                                        |
| ---------------------- | ---------------------------------------------- |
| `avg_warmup_iter_time` | Mean per-iteration time during warmup.         |
| `num_iters`            | Number of measured iterations.                 |
| `min_iter_time`        | Fastest measured iteration.                    |
| `max_iter_time`        | Slowest measured iteration.                    |
| `mean_iter_time`       | Mean measured iteration.                       |
| `median_iter_time`     | Median measured iteration.                     |
| `std_iter_time`        | Population standard deviation (0.0 if n == 1). |

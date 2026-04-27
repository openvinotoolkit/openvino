# Multithread Sync App (C++)

`multithread_sync_app` is a CPU-focused multi-thread synchronous inference sample.

## Purpose

- Create one compiled model on CPU.
- Create one infer request per application thread.
- Run synchronous `infer()` loops concurrently from multiple app threads.
- Measure throughput and per-iteration latency statistics (avg/p90/p99).

## Usage

```bash
./multithread_sync_app <model_path> <num_threads> \
	[-infer_precision <i8|u8|f32|bf16>] \
	[-shape [1,64]] \
	[-layout [NW]]
```

Example:

```bash
taskset -c 16,17,18,19 ./multithread_sync_app /path/to/model.xml 4 -shape [1,64] -layout [NW] -infer_precision INT8
```

## Notes

- The sample enables CPU property `ov::intel_cpu::multi_app_thread_sync_execution`.
- Intended for analyzing thread-to-stream behavior with synchronous inference on CPU.

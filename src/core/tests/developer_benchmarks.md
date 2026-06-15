# Developer Benchmarks

Developer-only benchmarks for comparing file-loading strategies. Use these to evaluate I/O
performance on new hardware or to validate changes to `ov::MappedMemory`.

These tests are **not compiled by default** — `ENABLE_DEVELOPER_TESTS` is OFF.

## Build

```bash
cmake -DENABLE_TESTS=ON -DENABLE_DEVELOPER_TESTS=ON -DCMAKE_BUILD_TYPE=Release <other flags> ..
cmake --build . --target ov_core_unit_tests
```

## Run

```bash
./ov_core_unit_tests --gtest_filter=*FileLoadBenchmark*
```

## Environment Requirements

For reliable cold-cache measurements the benchmark must be able to flush the
Linux page cache via `/proc/sys/vm/drop_caches`. This requires either:

- Running as **root**, or
- Running inside a **privileged container** (`docker run --privileged ...`)

Without this access the benchmark falls back to `posix_fadvise(DONTNEED)`,
which the kernel may silently ignore — results will be unreliable and a warning
is printed once at the start of the run.

## Available Tests

| Test | Description |
|------|-------------|
| `strategists_latency_and_throughput_table` | Compares different file loading strategies |
| `hint_prefetch_with_offset_table` | Measures partial-file prefetch at various offsets and region sizes |
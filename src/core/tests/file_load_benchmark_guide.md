# Developer Benchmarks

Developer-only benchmarks for comparing file-loading strategies. Use these to evaluate I/O
performance on new hardware or to validate changes to `ov::MappedMemory`.

These tests are **not compiled by default** — the target uses `EXCLUDE_FROM_ALL`.

## Build

```bash
cmake -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release <other flags> ..
cmake --build <dir> --target ov_file_load_benchmark
```

## Run

```bash
./ov_file_load_benchmark --gtest_filter=*FileLoadBenchmark*
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
| `strategies_read_memcpy` | Compares cold-cache load-and-copy performance across three strategies — `ifstream` read into a preallocated buffer, `mmap` + `memcpy`, and `mmap` + `hint_prefetch` + `memcpy`. Reports latency and throughput. |
| `strategies_mlock` | Measures the cost of making an entire file resident in memory without an additional user copy. |
| `hint_prefetch_with_offset_table` | Stresses partial-region `hint_prefetch` on a single 1200 MB file across a matrix of starting offsets and region sizes. Highlights alignment and offset effects on prefetch latency. |
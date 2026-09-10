# Developer Benchmarks

Developer-only benchmarks for comparing file-loading strategies. Use these to evaluate I/O
performance on new hardware or to validate changes to `ov::MappedMemory`.

These tests are **not compiled by default**. The executable is built only when the
developer option `ENABLE_TESTS_PER_SOURCE=ON` is set, and even then the target is
`EXCLUDE_FROM_ALL` — it must be built explicitly by name. `ENABLE_TESTS_PER_SOURCE` is a
developer-only flag and is not enabled in CI.

**Build in Release.** In a Debug build the file still compiles, but every `FileLoadBenchmark` test is **skipped**, because the measurements are meaningless without optimizations.

## Build

```bash
cmake -DENABLE_TESTS=ON -DENABLE_TESTS_PER_SOURCE=ON -DCMAKE_BUILD_TYPE=Release <other flags> ..
cmake --build <dir> --target ov_core_unit_tests_file_load_benchmark
```

## Run

```bash
./ov_core_unit_tests_file_load_benchmark
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
| `strategies_mlock` | Measures the cost of making an entire file resident in memory without an additional user copy. |
| `read_into_mmap_and_compute` | **compute scenario.** Compares a `std::transform` pass over the mapped bytes (mimicking a dequantization/dtype-conversion pass) with and without a preceding synchronous `hint_prefetch`, instead of `mlock()` or `memcpy()`. Files up to 10 GB. |
| `hint_prefetch_with_offset_table` | Stresses partial-region `hint_prefetch` on a single 1200 MB file across a matrix of starting offsets and region sizes. Highlights alignment and offset effects on prefetch latency. |


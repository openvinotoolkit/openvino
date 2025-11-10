GEMMV vs benchdnn — Repro Guide
================================

This folder contains:
- `gemmv_bench` — our single-thread GEMMV (N=1) micro-benchmark (u8×s8→fp32)
- `gemmv_vs_benchdnn.sh` — a convenience script to compare our kernel and oneDNN `benchdnn`

Our kernels auto-route purely by ISA (AMX / AVX-512 VNNI) and shape; no env “tuners” are required.

Build this subproject directly (from this folder):

```
cd src/plugins/intel_cpu/tools/gemmv_bench
cmake -S . -B build -G Ninja
cmake --build build --target gemmv_bench -j
# binary at build/bin/gemmv_bench
```

Build benchdnn
--------------

```
git clone https://github.com/oneapi-src/oneDNN.git onednn
cmake -S onednn -B onednn/build -DCMAKE_BUILD_TYPE=Release -DDNNL_BUILD_TESTS=ON -DDNNL_BUILD_EXAMPLES=OFF
cmake --build onednn/build --target benchdnn -j
# benchdnn binary: onednn/build/tests/benchdnn/benchdnn
```

Quick runs (compute-only, 1 thread)
-----------------------------------

Our bench (auto-route):
```
OMP_NUM_THREADS=1 build/bin/gemmv_bench 256 4096
```

Benchdnn (VNNI):
```
ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI OMP_NUM_THREADS=1 benchdnn --mode=p --matmul --dt=u8:s8:f32 256x4096:4096x1
```

Benchdnn (AMX) — if HW supports AMX:
```
OMP_NUM_THREADS=1 benchdnn --mode=p --matmul --dt=u8:s8:f32 256x4096:4096x1
```

Full comparison
---------------

From this folder run (auto-discovers gemmv_bench and benchdnn or set GEMMV_BIN/BENCHDNN_BIN):
```
src/plugins/intel_cpu/tools/gemmv_bench/gemmv_vs_benchdnn.sh
# CSV: writes to ./gemmv_vs_benchdnn_ext.csv and /tmp/gemmv_vs_benchdnn_ext.csv
```

Alternatively via CMake target from this folder:
```
cmake -S . -B build -G Ninja
cmake --build build --target compare -j
```

The script:
- runs benchdnn twice (VNNI-only and AMX) on a set of shapes
- runs our bench 3× (auto-route) and aggregates median
- prints a unified CSV to stdout and saves it to `/tmp/gemmv_vs_benchdnn_ext.csv`

Notes
-----
- Kernels auto-select AMX when available, otherwise AVX-512 VNNI. No environment flags are required for normal runs.
- ISA toggles are allowed for measurement only and honored by helper scripts:
  - `GEMMV_DISABLE_AMX=1` — skip AMX path (measure VNNI/JIT only).
  - `GEMMV_DISABLE_VNNI=1` — skip VNNI path (measure AMX or JIT only).

Sweep (per-ISA)
---------------
Use the helper to generate a CSV across all target shapes per ISA (AMX, VNNI, FP32 fallback):
```
src/plugins/intel_cpu/tools/gemmv_bench/sweep_isa_perf.sh > /tmp/gemmv_isa_sweep.csv
```
The CSV has columns: `suite,mode,kernel,M,K,time_ms,gflops`.
- All numbers are single-thread (OMP_NUM_THREADS=1) compute-only runs for reproducibility.

Analyze CSV
-----------
From this folder:
```
python3 analyze_csv.py gemmv_vs_benchdnn_ext.csv
```
Prints ours/bench ratios per shape and averages for benchdnn VNNI and AMX.

Modes (stability vs speed)
--------------------------
- Stable (default): the comparison script does not set `GEMMV_SKIP_*`; full selftests/calibration run inside `gemmv_bench` for reliability.
- Fast: set environment `GEMMV_FAST=1` to skip selftests/calibration and add a lightweight external warmup per run.
  Example: `GEMMV_FAST=1 src/plugins/intel_cpu/tools/gemmv_bench/gemmv_vs_benchdnn.sh`.

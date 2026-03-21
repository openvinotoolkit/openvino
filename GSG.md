# GSG: Build and Validation Guide for OpenVINO GPU Plugin BevPoolV2

This guide explains how to build OpenVINO and validate that the BevPoolV2 operator runs correctly in the Intel GPU plugin.

## 1. Build OpenVINO (Release + Tests)

Run from the OpenVINO repository root:

```bash
cd openvino
rm -rf build bin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build -j"$(nproc)"
```

Expected result:
- Build completes successfully.
- Test binaries are generated under bin/intel64/Release.

## 2. Functional Test for BevPoolV2

Build the GPU plugin target, then run BevPoolV2-related functional tests:

```bash
cd openvino
cmake --build build --target openvino_intel_gpu_plugin --parallel 8

./bin/intel64/Release/ov_gpu_func_tests --gtest_filter='*BevPoolV2*'
```

Expected result:
- BevPoolV2 test cases are discovered.
- All selected tests pass.

If no tests are shown, check test registration and filter string.

### 2.1 Accuracy acceptance thresholds (P1/P3 baseline)

Use the following thresholds for BevPoolV2 correctness acceptance:

- f32 baseline path: abs <= 1e-4, rel <= 1e-4
- f32 stress shapes with very large accumulation (for example M >= 300000): abs <= 3e-3, rel <= 3e-3
- f16 path: abs <= 2e-3, rel <= 2e-3

Rationale:
- f16 path may include mixed-precision accumulation/rounding effects.
- f32 path keeps a tighter bound to catch silent drift before optimization changes.

## 3. ONNX Runtime Validation with benchmark_app

Use the same ONNX model and shape settings on CPU and GPU, then compare behavior.

### 3.1 CPU run

```bash
cd openvino
./bin/intel64/Release/benchmark_app -m ./bevpool_v2_custom.onnx -d CPU -shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7314,3]" -niter 100 --nireq 1
```

### 3.2 GPU run

```bash
cd openvino
./bin/intel64/Release/benchmark_app -m ./bevpool_v2_custom.onnx -d GPU -shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7314,3]" -niter 100 --nireq 1
```

Expected result:
- Model is loaded successfully on both CPU and GPU.
- Inference runs to completion without runtime errors.
- Performance numbers (latency/throughput) are reported.

## 4. Quick Pass Criteria

BevPoolV2 is considered validated when all items below are true:
- OpenVINO build completes in Release mode with tests enabled.
- BevPoolV2 functional tests pass in ov_gpu_func_tests.
- benchmark_app runs successfully on both CPU and GPU with the same model and shape settings.

## 5. Parity and Error Statistics (Performance/Parity Baseline)

Run parity comparison against reference bins:

```bash
cd openvino
export PYTHONPATH=./bin/intel64/Release/python:$PYTHONPATH
export LD_LIBRARY_PATH=./bin/intel64/Release:$LD_LIBRARY_PATH
python compare_bevpool_ref.py --model ./bevpool_v2_custom.onnx --ref-dir ./scripts/bevpool_compare_inputs --topk 10
```

Notes:
- `compare_bevpool_ref.py` now maps model inputs by input names (feat/depth/indices/intervals), so the previous reshape-related `KeyError` from object-id mapping is removed.
- Ensure Python OpenVINO package and runtime libraries come from the same local build (`bin/intel64/Release`), otherwise parity may fail before inference.
- `scripts/bevpool_performance_and_parity_report.py` now auto-injects local `PYTHONPATH`/`LD_LIBRARY_PATH` for parity subprocesses by default.

Record these metrics in validation notes:

- max_abs
- mean_abs
- max_rel
- mean_rel

For later ref vs opt performance comparisons, keep a fixed table format:

| Variant | Device | Latency (ms) | Throughput (FPS) | max_abs | max_rel |
|---|---|---:|---:|---:|---:|
| ref | GPU | - | - | - | - |
| opt | GPU | - | - | - | - |
| ref | CPU | - | - | - | - |

### 5.1 One-command Performance/Parity Report Generation

You can generate the full performance/parity report (CPU/ref/opt benchmark + parity stats) with:

```bash
cd openvino
python scripts/bevpool_performance_and_parity_report.py \
  --repo-root . \
  --model ./bevpool_v2_custom.onnx \
  --ref-dir ./scripts/bevpool_compare_inputs \
  --shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7314,3]" \
  --benchmark-bin ./bin/intel64/Release/benchmark_app \
  --compare-script ./compare_bevpool_ref.py \
  --niter 100 \
  --nireq 1 \
  --topk 10 \
  --output ./bevpool_performance_parity_report.md
```

Optional overrides for custom build directories:

- `--ov-python-dir <path_to_local_python_pkg>` (default: `./bin/intel64/Release/python`)
- `--ov-lib-dir <path_to_local_runtime_libs>` (default: `./bin/intel64/Release`)

The script uses these runtime switches to control variant selection on GPU:

- `OV_GPU_BEVPOOL_V2_FORCE_REF=1` for ref row
- `OV_GPU_BEVPOOL_V2_FORCE_OPT8=1` for opt row

If a forced opt path is not supported on current hardware/runtime guards, GPU falls back to ref.

## 6. Common Issues

- Model parse failure:
  - Confirm bevpool_v2_custom.onnx is a valid ONNX file and not accidentally overwritten by text/log output.
- No BevPoolV2 tests matched:
  - Verify registration and test filter spelling: *BevPoolV2*.
- GPU runtime failure:
  - Rebuild openvino_intel_gpu_plugin and re-run functional tests first.

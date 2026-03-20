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

## 3. ONNX Runtime Validation with benchmark_app

Use the same ONNX model and shape settings on CPU and GPU, then compare behavior.

### 3.1 CPU run

```bash
cd openvino
./bin/intel64/Release/benchmark_app -m ./bevpool_v2_custom.onnx -d CPU -shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7313,3]" -niter 100 --nireq 1
```

### 3.2 GPU run

```bash
cd openvino
./bin/intel64/Release/benchmark_app -m ./bevpool_v2_custom.onnx -d GPU -shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7313,3]" -niter 100 --nireq 1
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

## 5. Common Issues

- Model parse failure:
  - Confirm bevpool_v2_custom.onnx is a valid ONNX file and not accidentally overwritten by text/log output.
- No BevPoolV2 tests matched:
  - Verify registration and test filter spelling: *BevPoolV2*.
- GPU runtime failure:
  - Rebuild openvino_intel_gpu_plugin and re-run functional tests first.

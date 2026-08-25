# src/core/tests — standalone smoke tests of the Vulkan core

The tests run the core **without openvino and without a Vulkan SDK**:
MSVC Build Tools + a local toolchain (`%LOCALAPPDATA%\vktools`) installed by
a single script.

## Contents

| File | Purpose |
|---|---|
| `CMakeLists.txt` | standalone test project: static core library + SPIR-V generation + one exe per `test_*.cpp`; Vulkan comes from the SDK or the vktools bootstrap |
| `build_tests.bat` | thin wrapper: `cmake -S . -B build` → `cmake --build` → run every test exe |
| `test_ops_smoke.cpp` | every `ir_op` against analytic references on both executors (CPU via `cpu_execute`, GPU via `vk_program`/`vk_network`), FB/PB round-trips (alpha/axis/perm/quant_constants), device-name dispatch (`CPU` / `GPU` / `GPU.N` / clean NPU error), `vk_available_devices()` |
| `test_nn_ops.cpp` | batched MatMul (+transpose_b), GELU, SwiGLU, conv bias contract |
| `test_shape_ops.cpp` | QuickGELU, RMSNorm, Pad, quantized batched MatMul, middle-axis Softmax, FB v5 round-trip |
| `test_attention.cpp` | scaled dot-product attention composed from core ops (scale → Q·Kᵀ → softmax → ·V with pairwise-batched MatMul) + Crop, both executors |
| `test_llm.cpp` | batch 6: causal Softmax, RoPE, KV-cache append, ArgMax, GQA (`Bb=1`), full decoder step from core ops |
| `test_perf.cpp` | tiled MatMul parity at tile boundaries + GPU throughput (GFLOP/s) |
| `test_optim.cpp` | graph passes: DCE, constant folding (cascade), transpose/act peepholes, numeric preservation |
| `test_safetensors.cpp` | writes a .safetensors file, reads it via `st_r`, drives a matmul from file weights |
| `tools/setup_vktools.ps1` | SDK-less toolchain bootstrap: glslang 16.5.0 + Vulkan-Headers 1.3.290 + generation of `vulkan-1.lib` from the system `C:\Windows\System32\vulkan-1.dll` (dumpbin → .def → lib) |

## Running

```bat
:: once (if no Vulkan SDK is installed)
powershell -ExecutionPolicy Bypass -File tools\setup_vktools.ps1

:: build + run
build_tests.bat
```

Expected result: every executable prints `ALL PASS`
(ops_smoke 37, nn_ops 10, shape_ops 12, attention 4, llm 12, perf parity+perf,
optim 9, safetensors 7 checks).

## Conventions

- A new test = a new `test_*.cpp` file in this folder: CMake builds it into
  `<name>.exe` and registers it with CTest automatically (`ctest --test-dir
  build`). The `test_` prefix is required.
- Build artifacts live in `build/` — ignored by the root `.gitignore`.
- Kernels are compiled from `../kernels/*.comp` on every build
  (`gen_spirv.cmake`) and embedded into `spirv_kernels.inc`.

## Notes for future tests

- The GGUF/Paddle reader tests lived in a temporary folder and did not
  survive its cleanup — they should be restored here as `test_gguf_fe.cpp`
  and `test_paddle_fe.cpp`, generating mini-models directly in code.

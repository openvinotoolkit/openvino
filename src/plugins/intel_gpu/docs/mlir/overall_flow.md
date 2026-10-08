# MLIR execution path in GPU plugin

## How to build and test

MLIR and Graph Compiler are not added as third-party submodules, so a suitable LLVM & Graph Compiler have to be
built manually and then passed to the OpenVINO build via `-DENABLE_MLIR_FOR_GPU=ON` + `GraphCompiler_DIR` /
`MLIR_DIR` / `LLVM_DIR`. At runtime the path is enabled with `OV_GPU_ENABLE_MLIR=1`
(`ov::intel_gpu::enable_mlir`), and is tested by the `tests/functional/mlir_op` suites, which are only built
when `ENABLE_MLIR_FOR_GPU=ON`.

Full instructions: [How to build and test OV with MLIR support](./build_and_test.md).

## Overview

The GPU plugin has an optional MLIR-based execution path for a subset of the model. A stage called
`transformMLIR` is added to the GPU's transformation pipeline that matches suitable subgraphs in
`ov::Model`, converts them to mlir-module(s) using [linalg-dialect](https://mlir.llvm.org/docs/Dialects/Linalg/),
and inserts an `ov::intel_gpu::op::MLIROp` operation representing the converted subgraph.

An actual compilation and execution of the converted MLIR module happens in a separate project called
`graph-compiler (GC)` - the project is "ingress-agnostic" and doesn't depend on OV specifically, it handles
an arbitrary mlir-linalg-code as an input and produces a GPU-binary combined with a cpu-side launching code.
The generated host code drives an abstract `gc::gpu::GpuRuntime` interface, which the plugin implements on top
of `cldnn::stream`/`cldnn::engine`, so the kernels are scheduled into the command list of the OV stream. The
OpenVINO side is only responsible for matching a suitable subgraph in `ov::Model`, converting it to MLIR as is
(all the optimizations are made on the graph-compiler side), and providing the runtime-info on inference
(buffers, dependencies, etc).

`MLIROp` naturally follows the OV's compile/infer semantic: on model compilation the MLIR module is fully
compiled to a binary (even if the module has dynamic shapes), on `infer()` it launches the compiled binary
(no extra latency).

## Communication with the graph-compiler

The MLIR/GC dependent code lives under `transformations/mlir` to avoid bringing mlir/gc includes to the main
plugin, and is reached through the MLIR-free `MLIRGpuProgram` in `transformations/mlir/interface/gpu_runtime.hpp`.
The simplified flow is following:

**A. Compilation:**

During `transformMLIR` each partitioned subgraph is lowered to an MLIR linalg module (using the shared
`MLIRContext`) and wrapped into an `MLIROp` that owns a `GcGpuProgram`. Its constructor submits the module to
`gc::gpu::GpuCompiler::compile()`, which **returns immediately** and compiles in the background, so the
subgraphs of a model are compiled in parallel. The compiler serializes the module and keeps the results in an
internal cache keyed by the source, the architecture and the compilation options.

The background compilation is joined at the end of the compilation phase, in `CreateMLIROp()`
(`plugin/ops/mlir_op.cpp`), when the `ov::Model` is converted into a `cldnn::program`: from that point on
`MLIROp` holds a ready `gc::gpu::Program` and the compilation errors are reported at `compile_model()` time
rather than at the first inference.

**B. Inference:**

The op becomes a `cldnn::mlir_primitive` that owns nothing but the `MLIROp` itself - there is no GPU kernel to
compile or cache for it. At inference `mlir_primitive_impl::execute_impl` calls `MLIRGpuProgram::execute()`,
which:

* stores one memref per input/output buffer via `Program::ArgsBuilder` (the USM pointer plus the shape and the
  dense row-major strides of the leading dimensions expected by the module);
* calls `gc::gpu::Program::main(args, runtime, deps, nDeps)`, passing the dependency events of the primitive;
* aggregates the events returned by the program into the event of the primitive.

The compiled program is shared by all the streams of the model. Each `cldnn::network` owns one `GcGpuRuntime`
bound to its stream, created through `MLIRGpuRuntime::create`. The plugin registers this factory through
`register_mlir_gpu_runtime()`.

## Code organization

All the MLIR/Graph-Compiler dependent sources live under `src/plugin/transformations/mlir/` and are built into a
dedicated OBJECT library `openvino_intel_gpu_mlir_obj` that alone gets the MLIR/GC include dirs and links
`GraphCompiler`; the object library is then linked into the plugin. This keeps `mlir/*.h` and `gc/*.h` out of
every other translation unit.

The headers `convert.hpp` and `gpu_runtime.hpp` in `transformations/mlir/interface/` are MLIR/GC free.
The same applies to the `MLIROp` (`include/intel_gpu/op/mlir_op.hpp`) and
`cldnn::mlir_primitive` (`include/intel_gpu/primitives/mlir_primitive.hpp`) declarations: no MLIR/GC types
cross this boundary, so the whole `graph` library stays MLIR-free.

## Feature enabling

The feature is disabled by default and gated twice:
* at **build time** by `-DENABLE_MLIR_FOR_GPU=ON` (default `OFF`) - when off, no MLIR related
  *implementations* are compiled (no pattern matching/conversion/unit tests/inference logic). The MLIR-related
  *definitions* are still included to the build though (`MLIROp` or `cldnn::mlir_primitive` header files) to
  avoid sudden broken includes. The option requires `ENABLE_GPU_DEBUG_CAPS` (i.e. `ENABLE_DEBUG_CAPS=ON`),
  since all the runtime knobs of the feature are DEBUG options and the path would otherwise be impossible
  to switch on.
* at **runtime** by `ov::intel_gpu::enable_mlir` property (env variable `OV_GPU_ENABLE_MLIR`) which is also
  `false` by default. The option is `DEBUG_GLOBAL`, i.e. it applies to all models of the process and is
  settable via the env variable only - neither the public API nor the GPU config file accept it.

## Supported subgraphs

`ScaledDotProductAttention` is the only operation that is enabled via MLIR path by default.

The MLIR path supports a lot more operations (see `transformations/mlir/common/converters`), there are unit
tests for them, but they were never tested on a "real model".

Enabling/disabling certain matching patterns can be controlled via the `ov::intel_gpu::mlir_patterns`
option (`OV_GPU_MLIR_PATTERNS` env variable):
* unset or empty - fall back to the default patterns (`sdpa=ScaledDotProductAttention`);
* `"*"` - enables conversion for every supported operation;
* `"name1=Type1,Type2;name2=Type3,Type4"` - match only the specified chains, e.g.
  `OV_GPU_MLIR_PATTERNS='mart=MatMul,Add,Reshape,Transpose;rms=Power,ReduceMean,Add,Sqrt,Divide'` would match
  projection subgraphs.

`mlir_patterns` is a per-model `DEBUG` option, so unlike `enable_mlir` it can also be set via the GPU
config file. Verbose logging of the pipeline is controlled by `ov::intel_gpu::mlir_debug`
(`OV_GPU_MLIR_DEBUG`), which is `DEBUG_GLOBAL` like `enable_mlir`.

## See also

 * [How to build and test OV with MLIR support](./build_and_test.md)
 * [OpenVINO GPU Plugin](../../README.md)
 * [Developer documentation](../../../../../docs/dev/index.md)

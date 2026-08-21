# MLIR execution path in GPU plugin

## How to build and test

MLIR and Graph Compiler are not added as third-party submodules, so a suitable LLVM & Graph Compiler have to be
built manually and then passed to the OpenVINO build via `-DENABLE_GPU_MLIR=ON` + `GraphCompiler_DIR` /
`MLIR_DIR` / `LLVM_DIR`. At runtime the path is enabled with `OV_GPU_ENABLE_MLIR=1`
(`ov::intel_gpu::enable_mlir`), and is tested by the `tests/functional/mlir_op` suites, which are only built
when `ENABLE_GPU_MLIR=ON`.

Full instructions: [How to build and test OV with MLIR support](./build_and_test.md).

## Overview

The GPU plugin has an optional MLIR-based execution path for a subset of the model. A stage called
`transformMLIR` is added to the GPU's transformation pipeline that matches suitable subgraphs in
`ov::Model`, converts them to mlir-module(s) using [linalg-dialect](https://mlir.llvm.org/docs/Dialects/Linalg/),
and inserts an `ov::intel_gpu::op::MLIROp` operation representing the converted subgraph.

An actual compilation and execution of the converted MLIR module happens in a separate project called
`graph-compiler (GC)` - the project is "ingress-agnostic" and doesn't depend on OV specifically, it handles
an arbitrary mlir-linalg-code as an input and produces a GPU-binary combined with a cpu-side launching code
(using OpenCL runtime). The OpenVINO side is only responsible for matching a suitable subgraph in `ov::Model`,
converting it to MLIR as is (all the optimizations are made on the graph-compiler side), and providing the
runtime-info on inference (OpenCL queue/context handlers, buffers, etc).

`MLIROp` naturally follows the OV's compile/infer semantic: on model compilation the MLIR module is fully
compiled to a binary (even if the module has dynamic shapes), on `infer()` it launches the compiled binary
(no extra latency).

## Communication with the graph-compiler

`MLIROp` communicates with the graph-compiler using its public api (the actual communication is delegated to a
separate `intel_gpu::mlir::MLIREvaluateGcGPU` class that is defined under `transformations/mlir` to avoid
bringing mlir/gc includes to the main plugin). The simplified flow is following:

**A. Compilation:**

During `transformMLIR` each partitioned subgraph is lowered to an MLIR linalg module and wrapped into an
`MLIROp`. The op's execution engine, `MLIREvaluateGcGPU`, is constructed first, and its constructor hands the
module to the Graph Compiler - `gc::gpu::OclModuleBuilder(module).build(device, context)` - which JIT-compiles
it and returns a `gc::gpu::OclModule`: a binary holding both the generated GPU kernels and the host code that
launches them.

By the time the `MLIROp` is inserted into the `ov::Model`, the subgraph it replaces is already fully compiled
into a device binary.

**B. Inference:**

The op becomes a `cldnn::mlir_primitive` that owns nothing but the `MLIROp` itself - there is no GPU kernel to
compile or cache for it. At inference `mlir_primitive_impl::execute_impl`:

* extracts the native OpenCL handles (`cl_mem` / USM pointers, `cl_command_queue`, dependency `cl_event`s) out
  of the cldnn runtime;
* passes them through an `ov::EvaluationContext` into `MLIROp::evaluate()`;
* turns the resulting `cl_event`s back into cldnn events.

The native handles are exposed by `get_native_handle()` accessors added to `cldnn::memory`, `cldnn::event` and
`cldnn::stream`. Only the OCL runtime implements them, so the MLIR path is OCL-only for now
(`mlir_primitive_impl` throws if a handle is unavailable).

## Code organization

All the MLIR/Graph-Compiler dependent sources live under `src/plugin/transformations/mlir/` and are built into a
dedicated OBJECT library `openvino_intel_gpu_mlir_obj` that alone gets the MLIR/GC include dirs and links
`GraphCompiler`; the object library is then linked into the plugin. This keeps `mlir/*.h` and `gc/*.h` out of
every other translation unit.

The only headers other plugin code may include are the ones from `transformations/mlir/interface/`
(`convert.hpp`, `mlir_evaluate_base.hpp`, `properties.hpp`) - they are MLIR/GC free. The same applies to the
`MLIROp` (`include/intel_gpu/op/mlir_op.hpp`) and `cldnn::mlir_primitive`
(`include/intel_gpu/primitives/mlir_primitive.hpp`) declarations: no MLIR/GC types cross this boundary, so the
whole `graph` library stays MLIR-free.

## Feature enabling

The feature is disabled by default and gated twice:
* at **build time** by `-DENABLE_GPU_MLIR=ON` (default `OFF`) - when off, no MLIR related
  *implementations* are compiled (no pattern matching/conversion/unit tests/inference logic). The MLIR-related
  *definitions* are still included to the build though (`MLIROp` or `cldnn::mlir_primitive` header files) to
  avoid sudden broken includes.
* at **runtime** by `ov::intel_gpu::enable_mlir` property (env variable `OV_GPU_ENABLE_MLIR`) which is also
  `false` by default. The option is `RELEASE_INTERNAL`, i.e. it is not settable via the public API -
  only via the env variable or the GPU config file (`ov::intel_gpu::config_file`).

## Supported subgraphs

`ScaledDotProductAttention` is the only operation that is enabled via MLIR path by default.

The MLIR path supports a lot more operations (see `transformations/mlir/common/converters`), there are unit
tests for them, but they were never tested on a "real model".

Enabling/disabling certain matching patterns can be controlled via `OV_MLIR_PATTERNS` env variable:
* unset - fall back to the default patterns (`sdpa=ScaledDotProductAttention`);
* empty string - enables conversion for every supported operation;
* `"name1=Type1,Type2;name2=Type3,Type4"` - match only the specified chains, e.g.
  `OV_MLIR_PATTERNS='mart=MatMul,Add,Reshape,Transpose;rms=Power,ReduceMean,Add,Sqrt,Divide'` would match
  projection subgraphs.

## See also

 * [How to build and test OV with MLIR support](./build_and_test.md)
 * [OpenVINO GPU Plugin](../../README.md)
 * [Developer documentation](../../../../../docs/dev/index.md)

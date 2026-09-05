# MLIR execution path: sources layout

This folder holds every MLIR / Graph-Compiler dependent source of the GPU plugin. It is compiled into a
dedicated `openvino_intel_gpu_mlir_obj` OBJECT library (only when `-DENABLE_GPU_MLIR=ON`), which is the single
place in the plugin allowed to include `mlir/*.h` and `gc/*.h`; the library is then linked into the plugin.

For *what* this path does and how it fits the plugin, see
[MLIR execution path in GPU plugin](../../../../docs/mlir/overall_flow.md). For how to get a build with it,
see [How to build and test OV with MLIR support](../../../../docs/mlir/build_and_test.md).

The entry point is `ov::intel_gpu::mlir::transformMLIR()`, declared in `interface/convert.hpp` and called from
the GPU transformation pipeline.

## Layout

| Path | Contents |
| --- | --- |
| `interface/` | The only headers the rest of the plugin should include. They are MLIR/GC-free. |
| `conversion/` | `ov::pass::MatcherPass` patterns, one per supported operation or class of operations. A pattern decides whether a node is eligible for the MLIR path (rank/precision/attribute checks) and attaches the converter that will lower it. |
| `common/` | Reusable conversion infrastructure shared by all converters: the per-operation converters themselves, the conversion context handed to them, and shape/type/constant helpers. See [common/README.md](common/README.md). |
| root `*.cpp` / `*.hpp` | The pass implementation: subgraph partitioning, whole-graph OV → MLIR-linalg conversion, and the Graph-Compiler-backed execution engine. |

## Adding support for a new operation

1. Add a converter under `common/converters/` following the interface described in
   [common/README.md](common/README.md).
2. Add a matcher pattern under `conversion/` that binds the new converter to the matched node, and register that
   pattern in the pass list built by `transformMLIR()`.
3. Add a functional test under `tests/functional/mlir_op/` (built only with `ENABLE_GPU_MLIR=ON`).

Note that being *supported* and being *enabled by default* are different things: the set of patterns applied by
default is narrow and can be overridden with the `OV_MLIR_PATTERNS` env variable
(see [overall_flow.md](../../../../docs/mlir/overall_flow.md)).

## See also

 * [MLIR execution path in GPU plugin](../../../../docs/mlir/overall_flow.md)
 * [How to build and test OV with MLIR support](../../../../docs/mlir/build_and_test.md)
 * [OV-nodes to MLIR-linalg converters](common/README.md)
 * [OpenVINO GPU Plugin](../../../../README.md)

# CPU Op Development Guide

How to add or extend an operation in the OpenVINO Intel CPU plugin. Work the four
phases in order — each produces what the next consumes. A reference-only or
portable-C++ op is finished after the node phase; an executor-based op continues
through optimization.

| Phase | Guide | Covers |
|---|---|---|
| 1. Analysis | [choosing_a_strategy.md](./choosing_a_strategy.md) | Locating the core op + reference, checking for an existing CPU node, the strategy decision matrix, supported precisions and layouts, CPU transformations, and the analysis-summary template. |
| 2. Implementation | [implementing_a_node.md](./implementing_a_node.md) | File layout and naming, `Type` enum + factory registration, node header/source skeletons, shape inference, mandatory dynamic-shape support, `OV_SWITCH` type dispatch, and the Eltwise fast path. |
| 3. Optimization | [executors_and_optimization.md](./executors_and_optimization.md) | The executor framework, `getImplementations` / `OV_CPU_INSTANCE_*`, concrete Ref/JIT/oneDNN executors, `CpuParallel`, JIT kernels + ISA targets, and eltwise emitters. (Skip for reference-only ops.) |
| 4. Testing | [testing_a_node.md](./testing_a_node.md) | Shared single-layer tests, custom CPU tests, the eltwise/activation reuse path, the edge-case matrix, ISA-selection checks, build/run commands, and troubleshooting. |

## See also

- [Selective build (Conditional Compilation)](../selective_build.md) — why `OV_SWITCH` and the `OV_CPU_INSTANCE_*` macros exist.
- [Internal CPU Plugin Optimizations](../internal_cpu_plugin_optimization.md) — graph-level fusions that may consume or rewrite an op before the node layer.
- [Convolution post ops](../convolution_post_ops.md) — post-op and zero-point mechanisms for oneDNN-backed ops.
- [JIT emitters](../../src/emitters/README.md) — emitter architecture, lifecycle, and debugging.
- [SIMD abstraction layer](../../src/nodes/kernels/simd/README.md) — `vec<T, isa>` for portable SIMD inside JIT kernels.
- [CPU plugin tests](../../tests/README.md) — functional-test directory taxonomy.
- [Debug capabilities](../debug_capabilities/README.md) — diagnosing a failing or slow op.
- [Coding style](../../../../../docs/dev/coding_style.md) — clang-format / clang-tidy / copyright rules and fix order.

# CPU Op Development Guide

How to add or extend an operation in the OpenVINO Intel CPU plugin. After
analysis, tests come **first**: write the single-layer tests against the op's
expected behaviour, then implement (and, if needed, optimize) the node until they
pass. A reference-only or portable-C++ op needs only the node phase; an
executor-based op also uses the executor framework.

| Phase | Guide | Covers |
|---|---|---|
| 1. Analysis | [choosing_a_strategy.md](./choosing_a_strategy.md) | Locating the core op + reference, checking for an existing CPU node, the strategy decision matrix, supported precisions and layouts, and CPU transformations. |
| 2. Tests | [testing_a_node.md](./testing_a_node.md) → [tests/README.md](../../tests/README.md) | Which two test files a new op needs; the shared + custom single-layer test patterns, edge cases, and ISA-selection checks live in the functional tests guide. |
| 3. Implementation | [implementing_a_node.md](./implementing_a_node.md) → [nodes/README.md](../../src/nodes/README.md) | Which files to create/update for the new op and the Eltwise fast path; the node lifecycle, registration, shape inference, and `OV_SWITCH` dispatch live in the nodes reference. |
| 4. Optimization | [executors_and_optimization.md](./executors_and_optimization.md) → [executors/README.md](../../src/nodes/executors/README.md) | What you create for your op (config, implementations, executors, wiring) and eltwise emitters; the executor framework, `CpuParallel`, JIT kernels, and oneDNN live in the executor reference. (Skip for reference-only ops.) |

## See also

- [Functional tests](../../tests/README.md) — how CPU functional tests are written and organized.
- [Nodes reference](../../src/nodes/README.md) — Node base-class lifecycle, registration, shape inference, `OV_SWITCH`.
- [Executor framework](../../src/nodes/executors/README.md) — `ExecutorImplementation`, `getImplementations`, `CpuParallel`, JIT/oneDNN executors.
- [Selective build (Conditional Compilation)](../selective_build.md) — `OV_SWITCH` and the `OV_CPU_INSTANCE_*` macros.
- [Internal CPU Plugin Optimizations](../internal_cpu_plugin_optimization.md) — graph-level fusions that may consume or rewrite an op before the node layer.
- [Convolution post ops](../convolution_post_ops.md) — post-op and zero-point mechanisms for oneDNN-backed ops.
- [JIT emitters](../../src/emitters/README.md) — emitter architecture, lifecycle, and debugging.
- [SIMD abstraction layer](../../src/nodes/kernels/simd/README.md) — `vec<T, isa>` for portable SIMD inside JIT kernels.
- [Debug capabilities](../debug_capabilities/README.md) — diagnosing a failing or slow op.
- [Coding style](../../../../../docs/dev/coding_style.md) — clang-format / clang-tidy / copyright rules and fix order.

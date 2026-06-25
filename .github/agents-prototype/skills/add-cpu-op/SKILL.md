---
name: add-cpu-op
description: >
  Add or extend an operation in the OpenVINO Intel CPU plugin — node registration,
  shape inference, executor/JIT/oneDNN implementations, and single-layer tests.
  Use when: a model fails on CPU with "not implemented" / unsupported-operation; a
  new core op needs CPU support; an existing CPU node needs a new precision, layout,
  or dynamic-shape support; or an op needs JIT/oneDNN optimization.
  Do not use for: adding an op to another plugin (GPU/NPU); core op (opset)
  definition or its reference implementation; graph transformations/fusions (those
  rewrite the graph before the node layer); or pure build/style/test-infra fixes
  with no new op.
---

# Add a CPU Op

Work the four phases in order. Each phase points to one reference doc under
`src/plugins/intel_cpu/docs/op_development/` — read the doc, do the work, move on.
Build after implementation and after optimization; do not advance on a broken build.

## 1. Analysis — pick a strategy
Read [choosing_a_strategy.md](../../../../src/plugins/intel_cpu/docs/op_development/choosing_a_strategy.md)
— it covers locating the core op + reference, checking for an existing CPU node,
the strategy decision matrix (reference-only / portable-C++ + CpuParallel /
executor-based), supported precisions and layouts, CPU transformations, and the
analysis-summary template. Produce the analysis summary; its `strategy` field
drives the rest.

## 2. Implementation — node + registration
Read [implementing_a_node.md](../../../../src/plugins/intel_cpu/docs/op_development/implementing_a_node.md)
— it covers file layout/naming, the `Type` enum + factory registration, the node
header/source skeletons, shape inference, mandatory dynamic-shape support,
`OV_SWITCH` type dispatch, and the Eltwise fast path for unary elementwise ops.
Build `openvino_intel_cpu_plugin` and confirm it compiles before continuing.

## 3. Optimization — executors / JIT / oneDNN  (skip if reference-only)
Skip this phase only when the analysis strategy is reference-only (see the skip
criteria in the doc); mark it `skipped` — overall status is still success if
phases 2 and 4 pass. Otherwise read
[executors_and_optimization.md](../../../../src/plugins/intel_cpu/docs/op_development/executors_and_optimization.md)
— it covers the executor framework, `getImplementations` / `OV_CPU_INSTANCE_*`,
concrete Ref/JIT/oneDNN executors, CpuParallel, JIT kernels + ISA targets, and
eltwise emitters. Rebuild and run `--gtest_filter=*<OpName>*` to confirm correctness.

## 4. Testing
Read [testing_a_node.md](../../../../src/plugins/intel_cpu/docs/op_development/testing_a_node.md)
— it covers shared single-layer tests, custom CPU tests, the eltwise/activation
reuse path, the edge-case matrix, ISA-selection checks, build/run commands, and
troubleshooting. Build `ov_cpu_func_tests`, run `*smoke*<OpName>*` and
`*<OpName>LayerCPUTest*`; all must pass.

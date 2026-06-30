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

Test-first workflow. After analysis, write the tests, then implement and optimize
the op until they pass. Each phase points to one reference doc under
`src/plugins/intel_cpu/docs/op_development/` (which links deeper into the CPU
plugin's general references). Read the doc, do the work, run the gate, move on.

## 1. Analysis — pick a strategy
Read [choosing_a_strategy.md](../../../../src/plugins/intel_cpu/docs/op_development/choosing_a_strategy.md)
— it covers locating the core op + reference, checking for an existing CPU node,
the strategy decision matrix (reference-only / portable-C++ + CpuParallel /
executor-based), precisions/layouts, and CPU transformations.

Produce this analysis summary and keep it for the rest of the run — its `strategy`
field decides whether phase 4 runs:

```
op_name:           <name>
opset_version:     <vX>
core_op_class:     <ov::op::vX::OpName>
reference_impl:    <path or "none">
existing_cpu_node: <path or "none">
strategy:          <reference-only | reference+parallel | jit-optimized | onednn-backed | executor-based>
inputs:            <list of {name, type, shape_info}>
outputs:           <list of {name, type, shape_info}>
attributes:        <list of {name, type, default}>
precisions:        <list of supported element types>
layouts:           <list of supported layout types>
dynamic_shapes:    <full | partial | static-only>
target_isa:        <list of ISA levels to target>
transformations:   <list of relevant CPU transformations or "none">
```

## 2. Tests first
Read [testing_a_node.md](../../../../src/plugins/intel_cpu/docs/op_development/testing_a_node.md)
(it points to the general [tests guide](../../../../src/plugins/intel_cpu/tests/README.md)).
Write the test files now — they encode the expected behaviour and will fail until
the op is implemented:
- shared: `tests/functional/shared_tests_instances/single_layer_tests/<op_name>.cpp`
- custom: `tests/functional/custom/single_layer_tests/<op_name>.cpp`
  (skip for the Eltwise fast path — extend the activation tests instead).
Cover the precisions, layouts, and static + dynamic shapes from the analysis summary.

## 3. Implementation — node + registration
Read [implementing_a_node.md](../../../../src/plugins/intel_cpu/docs/op_development/implementing_a_node.md)
(it points to the general [nodes reference](../../../../src/plugins/intel_cpu/src/nodes/README.md)
for lifecycle, registration, shape inference, dynamic shapes, and `OV_SWITCH`).
Create/register the node and the Eltwise fast path if applicable. Build and confirm
it compiles:

```bash
cd build && cmake --build . --target openvino_intel_cpu_plugin -j$(nproc) 2>&1 | tail -20
```

Iterate until the plugin builds and the phase-2 tests compile and link.

## 4. Optimization — executors / JIT / oneDNN  (skip if reference-only)
Skip this phase when the analysis `strategy` is reference-only (per the skip
criteria in the doc); mark it `skipped` — overall status is still success if
phases 2 and 3 pass. Otherwise read
[executors_and_optimization.md](../../../../src/plugins/intel_cpu/docs/op_development/executors_and_optimization.md)
(it points to the general [executor framework reference](../../../../src/plugins/intel_cpu/src/nodes/executors/README.md)).
Add the config, implementations, executors, and node wiring, then rebuild as above.

## 5. Verify
Build and run the tests; all must pass:

```bash
cd build && cmake --build . --target ov_cpu_func_tests -j$(nproc)
./bin/intel64/Release/ov_cpu_func_tests --gtest_filter=*smoke*<OpName>*
./bin/intel64/Release/ov_cpu_func_tests --gtest_filter=*<OpName>LayerCPUTest*
```

Report `success` with the list of files created/updated; if optimization was
skipped, note it.

# Implementing a CPU Node

This is the implementation phase of adding an operation to the Intel CPU plugin.
It assumes the analysis phase
([Choosing an Implementation Strategy](./choosing_a_strategy.md)) is done — op
name, strategy, precisions, layouts, and the shape-inference approach are known,
the core op class exists in `src/core/include/openvino/op/`, and a reference
implementation exists either in `src/core/reference/include/openvino/reference/`
or inside the core op's `evaluate()` method.

This page lists which files you create and update for a new op. For how a CPU node
is actually structured — the lifecycle methods, factory registration, shape
inference, dynamic shapes, `OV_SWITCH` type dispatch, and the header/source
skeleton — see the [nodes reference](../../src/nodes/README.md), which is the
source of truth for all of that. For ops that need JIT kernels, oneDNN primitives,
or ISA-specific paths, the executor framework is covered in
[Executors, Kernels and Optimization](./executors_and_optimization.md).

## Contents

- [Fast path: routing a unary elementwise op through Eltwise](#fast-path-routing-a-unary-elementwise-op-through-eltwise)
- [Files for a new node](#files-for-a-new-node)
- [Implementing the node](#implementing-the-node)

## Fast path: routing a unary elementwise op through Eltwise

If the new op is a **simple unary elementwise** op (one input tensor → one output
tensor, element-by-element, no attributes), the fastest path is to wire it through
the **existing `Eltwise` node** instead of creating a new CPU node class. This
avoids writing node boilerplate while still getting JIT emitter support and
eltwise fusion chains.

**Files to update (Eltwise path):**

| File | Change |
|------|--------|
| `src/plugins/intel_cpu/src/cpu_types.h` | Add `EltwiseOpName` to the `Algorithm` enum |
| `src/plugins/intel_cpu/src/cpu_types.cpp` | Add string name mapping in `algToString` |
| `src/plugins/intel_cpu/src/nodes/eltwise.cpp` | Map `ov::op::vX::OpName` → `Algorithm::EltwiseOpName` in `getAlgorithmFor`; add 1-input count entry |
| `src/plugins/intel_cpu/src/nodes/executors/ref/eltwise.cpp` | Add scalar reference case in the `switch` statement |
| `src/plugins/intel_cpu/src/post_ops.hpp` | Add `OpName` to `ActivationPostOp::Type` enum |
| `src/plugins/intel_cpu/src/post_ops.cpp` | Add bidirectional mapping `EltwiseOpName` ↔ `ActivationPostOp::Type::OpName` |
| `src/plugins/intel_cpu/src/nodes/kernels/x64/jit_uni_eltwise_generic.cpp` | Register new emitter class |
| `src/plugins/intel_cpu/src/nodes/kernels/aarch64/jit_uni_eltwise_generic.cpp` | Register new emitter class |
| `src/plugins/intel_cpu/src/nodes/kernels/riscv64/jit_uni_eltwise_generic.cpp` | Register new emitter class |
| `src/plugins/intel_cpu/src/emitters/snippets/x64/cpu_generator.cpp` | Register for snippets JIT |
| `src/plugins/intel_cpu/src/emitters/snippets/aarch64/cpu_generator.cpp` | Register for snippets JIT |
| `src/plugins/intel_cpu/src/emitters/snippets/riscv64/cpu_generator.cpp` | Register for snippets JIT |
| `src/plugins/intel_cpu/src/nodes/executors/jit/eltwise.cpp` | Remove op from the JIT exclusion list (if present) |

**JIT emitter files to create:**

| File | Change |
|------|--------|
| `src/plugins/intel_cpu/src/emitters/plugin/x64/jit_eltwise_emitters.hpp` | Declare `jit_<op_name>_emitter` class |
| `src/plugins/intel_cpu/src/emitters/plugin/x64/jit_eltwise_emitters.cpp` | Implement emitter + register table entries |
| `src/plugins/intel_cpu/src/emitters/plugin/aarch64/jit_eltwise_emitters.hpp` | Same for aarch64 |
| `src/plugins/intel_cpu/src/emitters/plugin/aarch64/jit_eltwise_emitters.cpp` | Same for aarch64 |
| `src/plugins/intel_cpu/src/emitters/plugin/riscv64/jit_eltwise_emitters.hpp` | Same for riscv64 |
| `src/plugins/intel_cpu/src/emitters/plugin/riscv64/jit_eltwise_emitters.cpp` | Same for riscv64 |

**Checking oneDNN post-op support:** Before implementing a JIT emitter, check if
oneDNN already supports the op as a post-op. If it does, the `post_ops.hpp/cpp`
wiring alone may be sufficient for post-op fusion chains.

The emitter classes themselves are written using the eltwise emitter pattern — see
[Eltwise JIT emitters](./executors_and_optimization.md#eltwise-jit-emitters) for
the emitter skeleton and coefficient tables, and the
[emitters README](../../src/emitters/README.md) for the emitter lifecycle and
debugging. Ops on this fast path do **not** need a separate custom test file — see
[the eltwise test path](../../tests/README.md#eltwise-routed-ops-the-activation-test-infrastructure).

## Files for a new node

All files follow **`snake_case`** for filenames, **`CamelCase`** for class names.
The build system uses `file(GLOB_RECURSE)` — new files under `src/` are
automatically picked up by CMake. No CMakeLists.txt edits needed for source files.

**Files to create:**

| File | Purpose |
|------|---------|
| `src/plugins/intel_cpu/src/nodes/<op_name>.h` | Node class header |
| `src/plugins/intel_cpu/src/nodes/<op_name>.cpp` | Node class implementation |

**Files to create (executor-based ops — any op more complex than portable C++):**

| File | Purpose |
|------|---------|
| `src/plugins/intel_cpu/src/nodes/executors/<op_name>_config.hpp` | `OpNameAttrs` struct + `OpNameConfig` alias |
| `src/plugins/intel_cpu/src/nodes/executors/<op_name>_implementations.cpp` | `getImplementations<OpNameAttrs>()` specialisation |
| `src/plugins/intel_cpu/src/nodes/executors/implementations.hpp` | (update) Add `getImplementations<OpNameAttrs>()` declaration |

See [Executors, Kernels and Optimization](./executors_and_optimization.md) for
how to fill these in.

**Files to update (factory registration):**

| File | Change |
|------|--------|
| `src/plugins/intel_cpu/src/cpu_types.h` | Add entry to `Type` enum |
| `src/plugins/intel_cpu/src/cpu_types.cpp` | Add string-to-Type mapping + `CASE` macro |
| `src/plugins/intel_cpu/src/nodes_factory.cpp` | Register node via `INTEL_CPU_NODE` macro |

The exact edits for these three files are described in
[Factory registration](../../src/nodes/README.md#factory-registration).

**Optional files (if needed):**

| File | When |
|------|------|
| `src/plugins/intel_cpu/src/shape_inference/custom/<op_name>.hpp` | Custom shape inference factory |
| `src/plugins/intel_cpu/src/shape_inference/custom/<op_name>.cpp` | Custom shape inference implementation |
| `src/plugins/intel_cpu/src/nodes/kernels/x64/<op_name>_kernel.hpp` | JIT kernel header (see optimization doc) |
| `src/plugins/intel_cpu/src/nodes/kernels/x64/<op_name>_kernel.cpp` | JIT kernel implementation (see optimization doc) |

## Implementing the node

Write the node header and source following the
[node skeleton](../../src/nodes/README.md#node-header-and-source-skeleton). The
points that most often need op-specific decisions:

- **Lifecycle methods** — which of `getSupportedDescriptors`,
  `initSupportedPrimitiveDescriptors`, `needPrepareParams`, `prepareParams`,
  `execute`, `executeDynamicImpl` to override, and what each does: see
  [Node lifecycle](../../src/nodes/README.md#node-lifecycle).
- **Shape inference** — `NgraphShapeInferFactory` (default) vs a custom factory:
  see [Shape inference factories](../../src/nodes/README.md#shape-inference-factories).
- **Dynamic shapes** — mandatory; see
  [Dynamic shapes](../../src/nodes/README.md#dynamic-shapes).
- **Multiple precisions** — dispatch with `OV_SWITCH`, never a manual `switch`:
  see [Type dispatch with OV_SWITCH](../../src/nodes/README.md#type-dispatch-with-ov_switch).

New files must follow the project coding standards — see
[Coding style](../../../../../docs/dev/coding_style.md).

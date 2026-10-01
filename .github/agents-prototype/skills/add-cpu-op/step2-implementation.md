# Skill: CPU Op Implementation

> Agent: `cpu_agent` — Step 2 of 4

## Prerequisites

- Completed **cpu_op_analysis** skill — op name, strategy, precisions, layouts,
  and shape inference approach are determined.
- Core op class exists in `src/core/include/openvino/op/`.
- Reference implementation exists in `src/core/reference/include/openvino/reference/`
  and/or inside the core op's `evaluate()` method.

## Fast Path: Eltwise Node (Unary Elementwise Ops)

If the new op is a **simple unary elementwise** op (one input tensor → one output tensor, element-by-element, no attributes), the fastest path is to wire it through the **existing `Eltwise` node** instead of creating a new CPU node class. This avoids writing node boilerplate while still getting JIT emitter support and eltwise fusion chains.

### Files to Update (Eltwise path)

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

### JIT Emitter Files to Create

| File | Change |
|------|--------|
| `src/plugins/intel_cpu/src/emitters/plugin/x64/jit_eltwise_emitters.hpp` | Declare `jit_<op_name>_emitter` class |
| `src/plugins/intel_cpu/src/emitters/plugin/x64/jit_eltwise_emitters.cpp` | Implement emitter + register table entries |
| `src/plugins/intel_cpu/src/emitters/plugin/aarch64/jit_eltwise_emitters.hpp` | Same for aarch64 |
| `src/plugins/intel_cpu/src/emitters/plugin/aarch64/jit_eltwise_emitters.cpp` | Same for aarch64 |
| `src/plugins/intel_cpu/src/emitters/plugin/riscv64/jit_eltwise_emitters.hpp` | Same for riscv64 |
| `src/plugins/intel_cpu/src/emitters/plugin/riscv64/jit_eltwise_emitters.cpp` | Same for riscv64 |

### Checking oneDNN Post-Op Support

Before implementing a JIT emitter, check if oneDNN already supports the op as a post-op. If it does, the `post_ops.hpp/cpp` wiring alone may be sufficient for post-op fusion chains.

## File Structure

All files follow **`snake_case`** for filenames, **`CamelCase`** for class names.
The build system uses `file(GLOB_RECURSE)` — new files under `src/` are
automatically picked up by CMake. No CMakeLists.txt edits needed for source files.

### Files to Create

| File | Purpose |
|------|---------|
| `src/plugins/intel_cpu/src/nodes/<op_name>.h` | Node class header |
| `src/plugins/intel_cpu/src/nodes/<op_name>.cpp` | Node class implementation |

### Files to Create (Executor-based ops — any op more complex than portable C++)

| File | Purpose |
|------|---------|
| `src/plugins/intel_cpu/src/nodes/executors/<op_name>_config.hpp` | `OpNameAttrs` struct + `OpNameConfig` alias |
| `src/plugins/intel_cpu/src/nodes/executors/<op_name>_implementations.cpp` | `getImplementations<OpNameAttrs>()` specialisation |
| `src/plugins/intel_cpu/src/nodes/executors/implementations.hpp` | (update) Add `getImplementations<OpNameAttrs>()` declaration |

### Files to Update

| File | Change |
|------|--------|
| `src/plugins/intel_cpu/src/cpu_types.h` | Add entry to `Type` enum |
| `src/plugins/intel_cpu/src/cpu_types.cpp` | Add string-to-Type mapping + `CASE` macro |
| `src/plugins/intel_cpu/src/nodes_factory.cpp` | Register node via `INTEL_CPU_NODE` macro |

### Optional Files (if needed)

| File | When |
|------|------|
| `src/plugins/intel_cpu/src/shape_inference/custom/<op_name>.hpp` | Custom shape inference factory |
| `src/plugins/intel_cpu/src/shape_inference/custom/<op_name>.cpp` | Custom shape inference implementation |
| `src/plugins/intel_cpu/src/nodes/kernels/x64/<op_name>_kernel.hpp` | JIT kernel header (Step 3) |
| `src/plugins/intel_cpu/src/nodes/kernels/x64/<op_name>_kernel.cpp` | JIT kernel implementation (Step 3) |

## Step-by-Step Implementation

1. Read [`src/nodes/README.md`](../../../../src/plugins/intel_cpu/src/nodes/README.md) —
   it covers the node lifecycle, factory registration (`Type` enum,
   `cpu_types.cpp` mapping, `nodes_factory.cpp`), shape inference factories,
   dynamic shapes, the node header/source skeleton, and the mandatory
   `OV_SWITCH` type-dispatch pattern.
2. Create/update the files listed in [File Structure](#file-structure) above,
   following that guide's patterns.

### Build Verification

```bash
cd build
cmake --build . --target ov_cpu_func_tests -j$(nproc) 2>&1 | tail -20
# Or for a quicker check:
cmake --build . --target openvino_intel_cpu_plugin -j$(nproc) 2>&1 | tail -20
```

Verify no compilation errors before proceeding.

## Executor Pattern (Standard Architecture for Non-Trivial Ops)

1. Read [`executors/README.md`](../../../../src/plugins/intel_cpu/src/nodes/executors/README.md) —
   it covers the `ExecutorFactory`/`ExecutorImplementation` architecture, the
   7 `ExecutorImplementation` fields, the `OV_CPU_INSTANCE_*` macros and
   `getImplementations` registration, the `Executor` base class and concrete
   (reference/JIT) executors, wiring a node to an `ExecutorFactory`, and the
   framework utilities.
2. Define the attrs struct, register the `getImplementations<OpNameAttrs>()`
   specialisation, implement the concrete executor classes, and wire the node
   to the `ExecutorFactory`, following that guide's patterns.

## Code Quality Checklist

Before proceeding to the next step, verify:

- [ ] `clang-format` passes: code follows `src/.clang-format` rules
  (Google style, 4-space indent, 120 col limit).
- [ ] `clang-tidy` passes: code follows `src/plugins/intel_cpu/src/.clang-tidy`
  rules.
- [ ] Copyright header is present on all new files.
- [ ] SPDX license identifier: `Apache-2.0`.
- [ ] Namespace is `ov::intel_cpu::node`.
- [ ] `[[nodiscard]]` on const getter methods.
- [ ] `[[maybe_unused]]` on `const dnnl::stream& strm` in `execute()` when
  the stream is not used.
- [ ] No raw `new` / `delete` — use smart pointers.
- [ ] No `using namespace std;` or similar broad using-directives.

## Output

- All source files created/updated per the file structure above.
- Build compiles without errors.
- Proceed to **cpu_op_optimization** skill.

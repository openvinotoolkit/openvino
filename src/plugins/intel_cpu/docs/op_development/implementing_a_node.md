# Implementing a CPU Node

This is the implementation phase of adding an operation to the Intel CPU plugin.
It assumes the analysis phase
([Choosing an Implementation Strategy](./choosing_a_strategy.md)) is done — op
name, strategy, precisions, layouts, and the shape-inference approach are known,
the core op class exists in `src/core/include/openvino/op/`, and a reference
implementation exists either in `src/core/reference/include/openvino/reference/`
or inside the core op's `evaluate()` method.

For ops that need JIT kernels, oneDNN primitives, or ISA-specific paths, this doc
covers the node skeleton and registration; the executor framework itself is
covered in [Executors, Kernels and Optimization](./executors_and_optimization.md).

## Contents

- [Fast path: routing a unary elementwise op through Eltwise](#fast-path-routing-a-unary-elementwise-op-through-eltwise)
- [File layout and naming conventions](#file-layout-and-naming-conventions)
- [Registering the node type](#registering-the-node-type)
- [The node header](#the-node-header)
- [The node source](#the-node-source)
- [Shape inference](#shape-inference)
- [Dynamic shapes (mandatory)](#dynamic-shapes-mandatory)
- [Type dispatch with OV_SWITCH](#type-dispatch-with-ov_switch)
- [Build verification](#build-verification)
- [Coding style](#coding-style)

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
[Testing a CPU Op](./testing_a_node.md#eltwise-routed-ops-reuse-the-activation-test-infrastructure).

## File layout and naming conventions

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

**Files to update:**

| File | Change |
|------|--------|
| `src/plugins/intel_cpu/src/cpu_types.h` | Add entry to `Type` enum |
| `src/plugins/intel_cpu/src/cpu_types.cpp` | Add string-to-Type mapping + `CASE` macro |
| `src/plugins/intel_cpu/src/nodes_factory.cpp` | Register node via `INTEL_CPU_NODE` macro |

**Optional files (if needed):**

| File | When |
|------|------|
| `src/plugins/intel_cpu/src/shape_inference/custom/<op_name>.hpp` | Custom shape inference factory |
| `src/plugins/intel_cpu/src/shape_inference/custom/<op_name>.cpp` | Custom shape inference implementation |
| `src/plugins/intel_cpu/src/nodes/kernels/x64/<op_name>_kernel.hpp` | JIT kernel header (see optimization doc) |
| `src/plugins/intel_cpu/src/nodes/kernels/x64/<op_name>_kernel.cpp` | JIT kernel implementation (see optimization doc) |

## Registering the node type

### 1. Add Type enum entry

In `src/plugins/intel_cpu/src/cpu_types.h`, add the new type to the `Type` enum:

```cpp
enum class Type : uint8_t {
    // ... existing entries ...
    OpName,      // <-- Add alphabetically or near related ops
};
```

### 2. Add string-to-Type mapping

In `src/plugins/intel_cpu/src/cpu_types.cpp`, add the mapping in TWO places:

**a) In the `type_to_name_tbl` map (op type string → Type enum):**

```cpp
{"OpName", Type::OpName},
```

**b) In the `type_to_str()` switch (Type enum → debug string), add a CASE:**

```cpp
CASE(OpName);
```

> **Note:** The string key in `type_to_name_tbl` must match the op's
> `get_type_name()` return value exactly. Check the core op's `OPENVINO_OP("...")`
> macro to find the correct string.

### 3. Register in the node factory

In `src/plugins/intel_cpu/src/nodes_factory.cpp`:

**a) Add the include:**

```cpp
#include "nodes/<op_name>.h"
```

**b) Add the registration macro in `NodesFactory::NodesFactory()`:**

```cpp
INTEL_CPU_NODE(OpName, Type::OpName);
```

> **Architecture-specific registration:** If the node is only supported on
> x86-64 (e.g., has JIT kernels), wrap both the include and registration in:
> ```cpp
> #if defined(OPENVINO_ARCH_X86_64)
>     INTEL_CPU_NODE(OpName, Type::OpName);
> #endif
> ```

## The node header

Create `src/plugins/intel_cpu/src/nodes/<op_name>.h`:

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class OpName : public Node {
public:
    OpName(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                     std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    [[nodiscard]] bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    // Op-specific attributes
    // Template helpers for type dispatch
};

}  // namespace ov::intel_cpu::node
```

**Key decisions for the header:**

| Question | If yes | If no |
|----------|--------|-------|
| Does the op need ISA-specific or multi-backend paths? | Use the Executor framework (`ExecutorFactory` + `ExecutorPtr`) — see [Executors, Kernels and Optimization](./executors_and_optimization.md) | Use direct `execute()` with `OV_SWITCH` type dispatch |
| Does the op need custom shape inference? | Add custom `ShapeInferFactory` | Use `NgraphShapeInferFactory` |
| Does the op need `prepareParams()`? | Override `needPrepareParams` → `true` | Override → `false` |
| Does the op have a data-dependent output shape? | Override `needShapeInfer()` | Don't override |

## The node source

Create `src/plugins/intel_cpu/src/nodes/<op_name>.cpp`:

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "<op_name>.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/<op_name>.hpp"              // Core op header
#include "openvino/reference/<op_name>.hpp"       // Reference implementation
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

// ═══════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════
OpName::OpName(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    // Extract attributes from the core op:
    // const auto typed_op = ov::as_type_ptr<const ov::op::vX::OpName>(op);
    // m_attribute = typed_op->get_attribute();
}

// ═══════════════════════════════════════════════════════════════════
// Supported Operation Check
// ═══════════════════════════════════════════════════════════════════
bool OpName::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                  std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::vX::OpName>(op)) {
            errorMessage = "Only opsetX OpName operation is supported";
            return false;
        }
        // Add additional checks: shape constraints, attribute values, etc.
    } catch (...) {
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// Descriptor Setup
// ═══════════════════════════════════════════════════════════════════
void OpName::getSupportedDescriptors() {
    // Validation is already done in the core op.
    // Add CPU-specific validation here if needed.
}

void OpName::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto inputPrecision = getOriginalInputPrecisionAtPort(0);
    auto outputPrecision = getOriginalOutputPrecisionAtPort(0);

    // Constrain to supported precisions
    if (none_of(inputPrecision,
                ov::element::f32,
                ov::element::bf16,   // requires avx512_core_bf16
                ov::element::f16,
                ov::element::i32)) {
        inputPrecision = ov::element::f32;
    }

    // Register supported primitive descriptors with layout+precision
    addSupportedPrimDesc(
        {{LayoutType::ncsp, inputPrecision}},   // inputs
        {{LayoutType::ncsp, outputPrecision}},   // outputs
        impl_desc_type::ref);                    // implementation type
}

// ═══════════════════════════════════════════════════════════════════
// Execution
// ═══════════════════════════════════════════════════════════════════
bool OpName::created() const {
    return getType() == Type::OpName;
}

bool OpName::needPrepareParams() const {
    return false;
}

void OpName::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void OpName::execute([[maybe_unused]] const dnnl::stream& strm) {
    // Option A: Call the reference implementation directly
    // ov::reference::op_name(getSrcDataAtPortAs<const T>(0),
    //                        getDstDataAtPortAs<T>(0),
    //                        ov::Shape{getSrcMemoryAtPort(0)->getStaticDims()},
    //                        /* attributes */);

    // Option B: Use OV_SWITCH for type dispatch
    // (see SearchSorted or SegmentMax for pattern)
}

}  // namespace ov::intel_cpu::node
```

## Shape inference

**Standard ops** — Use `NgraphShapeInferFactory(op)` in the constructor (default).
This delegates to the core op's `validate_and_infer_types()`.

**Custom shape inference** — When the core shape inference is insufficient or when
output shapes depend on input data at runtime:

Create `src/plugins/intel_cpu/src/shape_inference/custom/<op_name>.hpp`:

```cpp
#pragma once

#include <memory>
#include <utility>

#include "openvino/core/node.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

class OpNameShapeInferFactory : public ShapeInferFactory {
public:
    explicit OpNameShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};

}  // namespace ov::intel_cpu::node
```

Then in the node constructor, use `OpNameShapeInferFactory(op)` instead of
`NgraphShapeInferFactory(op)`.

## Dynamic shapes (mandatory)

Dynamic shapes are a **mandatory** requirement for new CPU node implementations.

Key methods to implement:

| Method | Purpose |
|--------|---------|
| `executeDynamicImpl(strm)` | Called during dynamic-shape execution. Usually delegates to `execute(strm)`. |
| `needPrepareParams()` | Return `true` if internal state (e.g., JIT kernel) must be rebuilt when shapes change. |
| `needShapeInfer()` | Override if output shape depends on input **data** (not just input shapes). |
| `createPrimitive()` | Called once after shapes are resolved. Allocate JIT kernels here (with caching). |

**Pattern for data-dependent output shapes** (see `SegmentMax` for reference):

```cpp
bool OpName::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    // Check if data that affects output shape has changed
    // Compare against cached values
    return false;
}
```

## Type dispatch with OV_SWITCH

For ops that need to handle multiple element types, **always** use the `OV_SWITCH`
macro for precision dispatch. This is **required** for the CPU plugin's
conditional compilation feature — `OV_SWITCH` allows the build system to
eliminate unused type specialisations at compile time, reducing binary size (see
[Selective build](../selective_build.md)).

**Do NOT** use manual `if/else` or `switch` chains on element type — they break
conditional compilation.

```cpp
namespace {
struct OpNameContext {
    OpName& node;
};
}  // namespace

template <class T>
struct OpName::OpNameExecute {
    void operator()(OpNameContext& ctx) {
        ctx.node.executeImpl<T>();
    }
};

void OpName::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto precision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    OpNameContext ctx = {*this};
    OV_SWITCH(intel_cpu,
              OpNameExecute,
              ctx,
              precision,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::bf16, ov::bfloat16),
              OV_CASE(ov::element::f16, ov::float16),
              OV_CASE(ov::element::i32, int32_t))
}
```

## Build verification

```bash
cd build
cmake --build . --target ov_cpu_func_tests -j$(nproc) 2>&1 | tail -20
# Or for a quicker check:
cmake --build . --target openvino_intel_cpu_plugin -j$(nproc) 2>&1 | tail -20
```

Verify no compilation errors before proceeding.

## Coding style

New files must follow the project coding standards (clang-format, clang-tidy,
copyright header with `Apache-2.0` SPDX identifier, the `ov::intel_cpu::node`
namespace, `[[nodiscard]]` / `[[maybe_unused]]` attributes, smart pointers instead
of raw `new`/`delete`, no broad `using namespace`). See
[Coding style](../../../../../docs/dev/coding_style.md) for the full rules and the
fix order.

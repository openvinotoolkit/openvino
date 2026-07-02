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

### 1. Add Type Enum Entry

In `src/plugins/intel_cpu/src/cpu_types.h`, add the new type to the `Type` enum:

```cpp
enum class Type : uint8_t {
    // ... existing entries ...
    OpName,      // <-- Add alphabetically or near related ops
};
```

### 2. Add String-to-Type Mapping

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

### 3. Create the Node Header

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
| Does the op need ISA-specific or multi-backend paths? | Use the Executor framework (`ExecutorFactory` + `ExecutorPtr`) — see section below | Use direct `execute()` with `OV_SWITCH` type dispatch |
| Does the op need custom shape inference? | Add custom `ShapeInferFactory` | Use `NgraphShapeInferFactory` |
| Does the op need `prepareParams()`? | Override `needPrepareParams` → `true` | Override → `false` |
| Does the op data-dependent output shape? | Override `needShapeInfer()` | Don't override |

### 4. Create the Node Source

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

### 5. Register in Node Factory

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

### 6. Shape Inference

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

### 7. Dynamic Shapes Support

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

### 8. Build Verification

```bash
cd build
cmake --build . --target ov_cpu_func_tests -j$(nproc) 2>&1 | tail -20
# Or for a quicker check:
cmake --build . --target openvino_intel_cpu_plugin -j$(nproc) 2>&1 | tail -20
```

Verify no compilation errors before proceeding.

## Type Dispatch Pattern (OV_SWITCH — Mandatory for Conditional Compilation)

For ops that need to handle multiple element types, **always** use the `OV_SWITCH`
macro for precision dispatch. This is **required** for the CPU plugin's
conditional compilation feature — `OV_SWITCH` allows the build system to
eliminate unused type specialisations at compile time, reducing binary size.

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

## Executor Pattern (Standard Architecture for Non-Trivial Ops)

The executor framework is the **standard architecture** for any CPU node that is
more complex than simple portable C++ code. This includes ops with:
- JIT kernels (these become `ExecutorType::Jit` implementations)
- oneDNN primitives (these become `ExecutorType::Dnnl` implementations)
- Multiple backend targets (x64/ARM/RISC-V)
- Any ISA-specific optimisation

Study these canonical implementations in this order:
1. `eltwise_implementations.cpp` — simplest executor pattern (Jit + ACL + Reference)
2. `fullyconnected_implementations.cpp` — full-featured (MLAS, 1x1conv, ACL, KleidiAI, oneDNN)
3. `convolution_implementations.cpp` — layout-aware implementations with `MemoryFormatFilter`

### Architecture Overview

```
Node (e.g. Eltwise, FullyConnected, Convolution)
 ├── OpNameAttrs           (op config struct: <op_name>_config.hpp)
 ├── MemoryArgs memory     (maps ARG_SRC/ARG_DST/... → MemoryPtr)
 ├── ExecutorFactoryPtr<OpNameAttrs> factory
 │    ├── filters ExecutorImplementation<Attrs> list by:
 │    │    ├── supports(config, memoryFormatFilter)   — ISA, precision, layout
 │    │    ├── createOptimalConfig(config)             — preferred memory format
 │    │    └── acceptsShape(attrs, memory)             — shape-dependent heuristics
 │    └── make(memory) → returns ExecutorPtr or VariableExecutor
 └── ExecutorPtr executor
      ├── update(memory)   — called in prepareParams()
      └── execute(memory)  — called in execute()
```

### Step 1: Define the Attrs Struct

Create `src/plugins/intel_cpu/src/nodes/executors/<op_name>_config.hpp`:

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"
#include "post_ops.hpp"  // if post-ops fusion is supported

namespace ov::intel_cpu {

struct OpNameAttrs {
    // Op-specific attributes needed by all implementations.
    // Examples: float epsilon; int axis; bool keep_dims;
    PostOps postOps;  // include if the op supports post-op fusion
};

using OpNameConfig = executor::Config<OpNameAttrs>;

}  // namespace ov::intel_cpu
```

### Step 2: Register the `getImplementations` Specialisation

**a) Declare in `implementations.hpp`** — add alongside existing declarations:

```cpp
// OpName
template <>
const std::vector<ExecutorImplementation<OpNameAttrs>>& getImplementations();
```

**b) Define in `<op_name>_implementations.cpp`:**

The file registers all implementation variants in **priority order** (highest
first). Each entry uses an `OV_CPU_INSTANCE_*` macro that conditionally compiles
the implementation based on the target architecture:

| Macro | Active When |
|-------|-------------|
| `OV_CPU_INSTANCE_COMMON(...)` | Always (portable code) |
| `OV_CPU_INSTANCE_X64(...)` | x86-64 only |
| `OV_CPU_INSTANCE_DNNL_X64(...)` | x86-64 with oneDNN |
| `OV_CPU_INSTANCE_DNNL(...)` | Any arch with oneDNN |
| `OV_CPU_INSTANCE_ACL(...)` | AArch64 with ARM Compute Library |
| `OV_CPU_INSTANCE_KLEIDIAI(...)` | AArch64 with KleidiAI |
| `OV_CPU_INSTANCE_MLAS_X64(...)` | x86-64 with MLAS |
| `OV_CPU_INSTANCE_RISCV64(...)` | RISC-V 64 only |

Each `ExecutorImplementation<Attrs>` has 6 fields:
1. **name** — unique human-readable string (e.g. `"op_name_jit_ncsp"`)
2. **ExecutorType** — `Jit`, `Dnnl`, `Acl`, `Reference`, etc.
3. **OperationType** — `OperationType::OpName`
4. **supports** — lambda returning `bool`. Use `VERIFY(condition, MESSAGE)` macro
   for debuggable rejection. Can take `(const Config&)` or
   `(const Config&, const MemoryFormatFilter&)`.
5. **createOptimalConfig** — lambda returning `std::optional<Config>`. Return
   `{}` (nullopt) if the current memory config is already acceptable. Use
   `createOptimalConfigCommon()` helper with `TypeMapping` and `LayoutConfig`.
6. **acceptsShape** — lambda `(const Attrs&, const MemoryArgs&) -> bool`. Pass
   `AcceptsAnyShape<Attrs>` (or `{}`) for shape-agnostic implementations.
7. **create** — lambda returning `ExecutorPtr`. Use `CreateDefault<Executor, Attrs>{}`
   for simple cases or a custom lambda for complex wiring (e.g., `DnnlExecutor`).

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/<op_name>_config.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "utils/arch_macros.h"

// Include the actual executor headers:
#include "nodes/executors/jit/<op_name>_jit.hpp"      // JIT executor (x64)
// #include "nodes/executors/acl/<op_name>_acl.hpp"    // ACL executor (ARM)
#include "nodes/executors/ref/<op_name>_ref.hpp"       // Reference executor

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

// TypeMapping defines precision transformations per implementation.
// Each row: {input_precisions..., output_precisions...} → {transform_functions...}
// Transform helpers: bypass() = keep as-is, just<T>() = force to T, use<N>() = match arg N
static const TypeMapping opNameTypeMapping {
    // {src, dst}            pt<src, dst>
    {{_f32, _f32},           {bypass(), bypass()}},
    {{_bf16, _bf16 | _f32},  {bypass(), bypass()}},
    {{_any, _any},           {just<f32>(), just<f32>()}},  // fallback
};

static const MappingNotation opNameMappingNotation {
    {ARG_SRC, 0},
    {ARG_DST, 1},
};

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<OpNameAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<OpNameAttrs>> implementations {
        OV_CPU_INSTANCE_X64(
            "op_name_jit_ncsp",
            ExecutorType::Jit,
            OperationType::OpName,
            [](const OpNameConfig& config) -> bool {
                VERIFY(dnnl::impl::cpu::x64::mayiuse(
                    dnnl::impl::cpu::x64::avx2), UNSUPPORTED_ISA);
                // Add precision / attr checks as needed
                return true;
            },
            // createOptimalConfig
            [](const OpNameConfig& config) -> std::optional<OpNameConfig> {
                return createOptimalConfigCommon(config,
                    opNameTypeMapping,
                    LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                    opNameMappingNotation);
            },
            AcceptsAnyShape<OpNameAttrs>,
            CreateDefault<JitOpNameExecutor, OpNameAttrs>{}
            )
        OV_CPU_INSTANCE_COMMON(
            "op_name_ref_ncsp",
            ExecutorType::Reference,
            OperationType::OpName,
            [](const OpNameConfig& config) -> bool { return true; },
            [](const OpNameConfig& config) -> std::optional<OpNameConfig> {
                return createOptimalConfigCommon(config,
                    opNameTypeMapping,
                    LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                    opNameMappingNotation);
            },
            AcceptsAnyShape<OpNameAttrs>,
            CreateDefault<RefOpNameExecutor, OpNameAttrs>{}
            )
    };
    return implementations;
}
// clang-format on

}  // namespace ov::intel_cpu
```

### Step 3: Implement the Concrete Executor Classes

Each executor implements the `Executor` base class from `executor.hpp`:

```cpp
class Executor {
public:
    virtual bool update(const MemoryArgs& memory) = 0;   // reshape / re-init
    virtual void execute(const MemoryArgs& memory) = 0;   // run computation
    virtual impl_desc_type implType() const = 0;           // report impl type
    virtual ~Executor() = default;
};
```

**Reference executor** (portable C++):

```cpp
class RefOpNameExecutor : public Executor {
public:
    RefOpNameExecutor(const OpNameAttrs& attrs,
                      const MemoryArgs& memory,
                      const ExecutorContext::CPtr& context);
    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    impl_desc_type implType() const override { return impl_desc_type::ref; }
};
```

**JIT executor** (wraps JIT kernel + CpuParallel):

```cpp
class JitOpNameExecutor : public Executor {
public:
    JitOpNameExecutor(const OpNameAttrs& attrs,
                      const MemoryArgs& memory,
                      const ExecutorContext::CPtr& context);
    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    impl_desc_type implType() const override { return impl_desc_type::jit_avx2; }
private:
    std::shared_ptr<JitKernelBase> m_kernel;
    // ... kernel compile params, context, etc.
};
```

### Step 4: Wire the Node to Use ExecutorFactory

In the **node header**, replace direct `execute()` logic with executor fields:

```cpp
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/<op_name>_config.hpp"
#include "nodes/executors/memory_arguments.hpp"

// Inside the class:
private:
    OpNameAttrs m_attrs;
    MemoryArgs m_memory;
    ExecutorFactoryPtr<OpNameAttrs> m_factory;
    ExecutorPtr m_executor = nullptr;
```

In **`initSupportedPrimitiveDescriptors()`** (follows `conv.cpp` pattern):

```cpp
void OpName::initSupportedPrimitiveDescriptors() {
    // 1. Populate m_attrs from the core op
    // m_attrs.epsilon = ...; m_attrs.axis = ...;
    // m_attrs.postOps = getPostOps(fusedWith);

    // 2. Build memory descriptors per port
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    VecMemoryDescs srcDescs;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        srcDescs.push_back(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            getOriginalInputPrecisionAtPort(i), getInputShapeAtPort(i)));
    }
    auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0));

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[0]},
        {ARG_DST, dstDesc},
    };

    // 3. Create executor factory — this filters & ranks implementations
    auto executionContext = std::make_shared<ExecutorContext>(
        context, getImplPriority());
    m_factory = std::make_shared<ExecutorFactory<OpNameAttrs>>(
        m_attrs, executionContext, descs);

    // 4. Let factory decide optimal memory descriptors
    const auto nodeDescList = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& nodeDescs : nodeDescList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(srcDescs.size());
        // ... populate from nodeDescs (see conv.cpp / fc.cpp pattern) ...
        nodeConfig.outConfs.emplace_back(nodeDescs.at(ARG_DST));
        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}
```

In **`prepareParams()`**:

```cpp
void OpName::prepareParams() {
    m_memory[ARG_SRC] = getSrcMemoryAtPort(0);
    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    if (!m_executor) {
        m_executor = m_factory->make(m_memory);
    }
    m_executor->update(m_memory);
}
```

In **`execute()`**:

```cpp
void OpName::execute([[maybe_unused]] const dnnl::stream& strm) {
    m_executor->execute(m_memory);
}
```

### Key Executor Framework Utilities

| Utility | Location | Purpose |
|---------|----------|---------|
| `VERIFY(cond, msg)` | `debug_messages.hpp` | Debug-logged rejection in `supports()` |
| `TypeMapping` / `MappingNotation` | `precision_translation.hpp` | Precision transform rules |
| `createOptimalConfigCommon()` | `implementation_utils.hpp` | Standard config optimisation |
| `AcceptsAnyShape<Attrs>` | `implementation_utils.hpp` | Shape-agnostic stub |
| `HasNoOptimalConfig<Attrs>` | `implementation_utils.hpp` | No config override stub |
| `SupportsAnyConfig<Attrs>` | `implementation_utils.hpp` | Unconditional support stub |
| `CreateDefault<Executor, Attrs>` | `implementation_utils.hpp` | Simple `make_shared` factory |
| `CreateDnnlDefault<Prim, Attrs>` | `implementation_utils.hpp` | `DnnlExecutor` wrapper factory |
| `OV_CPU_INSTANCE_*` macros | `utils/arch_macros.h` | Conditional compilation per arch |

> **Key advantage:** The `ExecutorFactory` automatically handles runtime
> selection between JIT / oneDNN / ACL / Reference implementations based on
> hardware capabilities, precision matching, and memory format preferences.
> When multiple implementations match, `VariableExecutor` handles fallback.
> The `OV_CPU_INSTANCE_*` macros ensure that code for unavailable platforms is
> compiled away, supporting the conditional compilation infrastructure.

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

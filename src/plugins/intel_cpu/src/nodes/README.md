# CPU Plugin Nodes

A *node* is the CPU plugin's executable representation of an OpenVINO operation.
Every node derives from the `Node` base class ([node.h](../node.h)) and is created
by the node factory ([nodes_factory.cpp](../nodes_factory.cpp)) from an
`ov::Node`. This document is the reference for how a CPU node is structured — its
lifecycle methods, factory registration, shape inference, and type dispatch. It is
general background for anyone working on CPU nodes; for the workflow of adding a
*new* op see [docs/op_development](../../docs/op_development/README.md).

## Contents

- [Node lifecycle](#node-lifecycle)
- [Factory registration](#factory-registration)
- [Shape inference factories](#shape-inference-factories)
- [Dynamic shapes](#dynamic-shapes)
- [Type dispatch with OV_SWITCH](#type-dispatch-with-ov_switch)
- [Node header and source skeleton](#node-header-and-source-skeleton)

## Node lifecycle

The `Node` base class declares the virtual methods a concrete node overrides. They
are called by the graph in roughly this order — once at compile time
(descriptors), then per inference (params + execute):

| Method | When called | Purpose |
|--------|-------------|---------|
| constructor `OpName(op, context)` | graph build | Validate the op via `isSupportedOperation`, extract attributes, pass a shape-infer factory to the `Node` base. |
| `getSupportedDescriptors()` | compile | Pure virtual. CPU-specific validation / descriptor preparation (often empty when the core op already validated). |
| `initSupportedPrimitiveDescriptors()` | compile | Register the supported `{layout, precision}` combinations per port via `addSupportedPrimDesc`, or build them from an `ExecutorFactory`. |
| `created()` | compile | Pure virtual. Return `getType() == Type::OpName`. |
| `needPrepareParams()` | compile/runtime | Return `true` if internal state (e.g. a JIT kernel) must be rebuilt when shapes change. |
| `createPrimitive()` | after shapes resolved | Allocate primitives/kernels once shapes are known (with caching). |
| `prepareParams()` | per inference (if `needPrepareParams`) | Refresh memory pointers and `update()` the executor. |
| `execute(strm)` | per inference (static) | Pure virtual. Run the computation. |
| `executeDynamicImpl(strm)` | per inference (dynamic) | Run under dynamic shapes; usually delegates to `execute(strm)`. |
| `needShapeInfer()` | per inference | Override if the output shape depends on input **data**, not just input shapes. |

`isSupportedOperation(op, errorMessage)` is a `static noexcept` method used both by
the constructor and by the graph to decide whether the node can handle a given
`ov::Node`. It must never throw.

## Factory registration

A node type is registered in three places so the factory can instantiate it from an
`ov::Node`:

1. **`Type` enum** in [cpu_types.h](../cpu_types.h) — add `OpName`.
2. **String ↔ `Type` mapping** in [cpu_types.cpp](../cpu_types.cpp): add
   `{"OpName", Type::OpName}` to `type_to_name_tbl`, and a `CASE(OpName)` in the
   `type_to_str()` switch. The string key must match the core op's
   `get_type_name()` (the `OPENVINO_OP("...")` macro) exactly.
3. **Factory entry** in [nodes_factory.cpp](../nodes_factory.cpp): include the node
   header and add `INTEL_CPU_NODE(OpName, Type::OpName);`. For nodes that only
   build on x86-64 (e.g. JIT-only), guard both with
   `#if defined(OPENVINO_ARCH_X86_64)`.

## Shape inference factories

The `Node` constructor takes a shape-inference factory:

- **`NgraphShapeInferFactory(op)`** — the default. Delegates to the core op's
  `validate_and_infer_types()`. Use it whenever the core shape inference is
  sufficient.
- **Custom `ShapeInferFactory`** — when the core inference is insufficient or the
  output shape depends on runtime data. Implemented under
  [shape_inference/custom/](../shape_inference/custom/) as an
  `OpNameShapeInferFactory : public ShapeInferFactory` whose `makeShapeInfer()`
  returns the op-specific `ShapeInferPtr`.

## Dynamic shapes

Dynamic shapes are mandatory for CPU nodes. The relevant overrides:

- `executeDynamicImpl(strm)` — entry point under dynamic shapes; delegate to
  `execute(strm)` unless the dynamic path differs.
- `needPrepareParams()` — `true` when shape changes require rebuilding kernels.
- `needShapeInfer()` — override only for **data-dependent** output shapes; compare
  the relevant inputs against cached values and return `true` when they changed
  (see `SegmentMax` for an example).
- `createPrimitive()` — allocate the kernel after shapes are resolved.

```cpp
bool OpName::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    // Output shape depends on input data: compare against cached values.
    return false;
}
```

## Type dispatch with OV_SWITCH

For nodes that handle multiple element types, **always** dispatch with the
`OV_SWITCH` macro rather than a manual `if/else` or `switch` on the element type.
`OV_SWITCH` lets conditional compilation eliminate unused type specialisations at
compile time, reducing binary size; manual chains defeat it. See
[selective_build.md](../../docs/selective_build.md#in-code-conditional-compilation)
for the macro reference.

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

## Node header and source skeleton

Files use `snake_case` names and `CamelCase` classes, in the
`ov::intel_cpu::node` namespace. The build uses `file(GLOB_RECURSE)`, so new files
under `src/` are picked up automatically — no CMakeLists edits.

```cpp
// <op_name>.h
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
    // Op-specific attributes; template helpers for type dispatch.
};

}  // namespace ov::intel_cpu::node
```

```cpp
// <op_name>.cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "<op_name>.h"

#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/op/<op_name>.hpp"          // core op
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

OpName::OpName(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    // const auto typed_op = ov::as_type_ptr<const ov::op::vX::OpName>(op);
    // m_attribute = typed_op->get_attribute();
}

bool OpName::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                  std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::vX::OpName>(op)) {
            errorMessage = "Only opsetX OpName operation is supported";
            return false;
        }
        // Additional checks: shape constraints, attribute values, etc.
    } catch (...) {
        return false;
    }
    return true;
}

void OpName::getSupportedDescriptors() {
    // CPU-specific validation if needed (core op already validated).
}

void OpName::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    auto inputPrecision = getOriginalInputPrecisionAtPort(0);
    auto outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (none_of(inputPrecision, ov::element::f32, ov::element::bf16,
                ov::element::f16, ov::element::i32)) {
        inputPrecision = ov::element::f32;
    }
    addSupportedPrimDesc({{LayoutType::ncsp, inputPrecision}},    // inputs
                         {{LayoutType::ncsp, outputPrecision}},   // outputs
                         impl_desc_type::ref);
}

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
    // Option A: call the reference implementation directly.
    // Option B: dispatch element type with OV_SWITCH (see above).
}

}  // namespace ov::intel_cpu::node
```

For nodes more complex than portable C++ (JIT, oneDNN, multi-ISA), the body of
`initSupportedPrimitiveDescriptors`/`prepareParams`/`execute` is driven by the
executor framework — see [executors/README.md](executors/README.md).

## See also

- [executors/README.md](executors/README.md) — executor framework for non-trivial ops.
- [kernels/simd/README.md](kernels/simd/README.md) — SIMD abstraction for kernels.
- [../../docs/selective_build.md](../../docs/selective_build.md) — `OV_SWITCH` / `OV_CPU_INSTANCE_*` and conditional compilation.
- [../../docs/op_development](../../docs/op_development/README.md) — adding a new CPU op.

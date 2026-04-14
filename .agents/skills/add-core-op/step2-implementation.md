# Skill: Core Op Implementation

> Source: `skills/add-core-op/SKILL.md` (Step 2)
> Agent: `core_opspec_agent`

## Prerequisites

- Completed **core_op_analysis** skill - op name, target opset, inputs,
  outputs, attributes are known.

## Files to Create

| File | Purpose |
|------|---------|
| `openvino/src/core/include/openvino/op/<op_name>.hpp` | Class declaration |
| `openvino/src/core/src/op/<op_name>.cpp` | Implementation |
| `openvino/src/core/reference/include/openvino/reference/<op_name>.hpp` | Reference kernel |
| `openvino/src/core/shape_inference/include/<op_name>_shape_inference.hpp` | Shape inference (if needed) |

> **Note:** If adding a new version of an existing op, some files already exist -
> only create what's missing.

## Files to Update

| File | Change |
|------|--------|
| `openvino/src/core/dev_api/openvino/op/ops_decl.hpp` | Register new Op class |
| `openvino/src/core/include/openvino/op/ops.hpp` | Include new op header |
| `openvino/src/core/include/openvino/opsets/opsetX_tbl.hpp` | Register in the latest opset table |

## Class Structure

### Header (`<op_name>.hpp`)

```cpp
#pragma once
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace vX {

class OPENVINO_API OpName : public Op {
public:
    OPENVINO_OP("OpName", "opsetX");

    /// \brief Default constructor (required for deserialization)
    OpName() = default;

    /// \brief Constructs OpName from inputs and attributes
    OpName(const Output<Node>& input, /* attributes */);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(
        const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs,
                  const TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    // attributes as members
};

}  // namespace vX
}  // namespace op
}  // namespace ov
```

> **Important:** Only add `evaluate()` / `has_evaluate()` to the core op class if the op
> is on the **constant-folding path** (e.g. shape-related ops with constant inputs) OR
> as a fallback when no plugin provides native execution. For ops with full CPU plugin
> support, move `evaluate()` to the **Template plugin** instead — see Template Plugin
> Integration below. This keeps the core library binary smaller.

**Notes:**
- Namespace: `ov::op::vX` - use the latest opset version number for X.
- Base class: use `Op` by default. Use a utility base when appropriate
  (e.g. `util::UnaryElementwiseArithmetic` for unary elementwise ops).
- Add `OPENVINO_OP("OpName", "opsetX")` inside the class body. If using a
  utility base, pass the base type as third argument - see existing ops.

### Source (`<op_name>.cpp`)

```cpp
#include "openvino/op/<op_name>.hpp"
#include "itt.hpp"
#include "element_visitor.hpp"
#include "openvino/reference/<op_name>.hpp"

namespace ov {
namespace op {
namespace vX {

OpName::OpName(const Output<Node>& input /*, attributes */)
    : Op({input}) /*, m_attr(attr) */ {
    OV_OP_SCOPE(vX_OpName_OpName);
    constructor_validate_and_infer_types();
}

void OpName::validate_and_infer_types() {
    OV_OP_SCOPE(vX_OpName_validate_and_infer_types);
    // Validate input shapes and types
    // Set output shapes and element types
}

bool OpName::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(vX_OpName_visit_attributes);
    // visitor.on_attribute("attr_name", m_attr);
    return true;
}

std::shared_ptr<Node> OpName::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(vX_OpName_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<OpName>(new_args.at(0) /*, m_attr */);
}

bool OpName::evaluate(TensorVector& outputs,
                      const TensorVector& inputs) const {
    OV_OP_SCOPE(vX_OpName_evaluate);
    // IF_TYPE_OF_CONVERT_TENSORS calls ov::reference::<op_name>
    return true;
}

bool OpName::has_evaluate() const {
    OV_OP_SCOPE(vX_OpName_has_evaluate);
    // return true for supported element types
    return true;
}

}  // namespace vX
}  // namespace op
}  // namespace ov
```

### Key Requirements

- **Conditional compilation:** Add `OV_OP_SCOPE(vX_OpName_method)` at the start
  of every method in the `.cpp` file.
- **Shape inference:** If complex, implement in a separate `shape_infer` function
  in `openvino/src/core/shape_inference/include/`. Use `PartialShape` to handle
  static and dynamic shapes.
- **`validate_and_infer_types`:** Validate inputs, then call shape inference.
  If an input is a `Constant` (e.g. axis), its value can be read at compile time.

## Template Plugin Integration

For ops that have full plugin support (native CPU/GPU execution), `evaluate()` lives in
the Template plugin, not in the core op class. This keeps the core binary small.

### Files to Create/Update

| File | Change |
|------|--------|
| `openvino/src/plugins/template/backend/ops/<op_name>.cpp` | `evaluate_node<ov::op::vX::OpName>` specialization |
| `openvino/src/plugins/template/backend/ops/ops_evaluates.hpp` | Add `extern template bool evaluate_node<ov::op::vX::OpName>` |
| `openvino/src/plugins/template/backend/opset_int_tbl.hpp` | Add `_OPENVINO_OP_REG(OpName, ov::op::vX)` (alphabetically) |

No CMakeLists change needed — the backend uses `file(GLOB ...)` to pick up all `.cpp` in `ops/`.

### Pattern for `<op_name>.cpp`

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/reference/<op_name>.hpp"

#include "evaluate_node.hpp"
#include "openvino/op/<op_name>.hpp"

namespace {
template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::vX::OpName>& node,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::reference::op_name<T>(inputs[0].data<const T>(),
                              outputs[0].data<T>(),
                              ov::shape_size(inputs[0].get_shape()));
    return true;
}
}  // namespace

template <>
bool evaluate_node<ov::op::vX::OpName>(std::shared_ptr<ov::Node> node,
                                       ov::TensorVector& outputs,
                                       const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(as_type_ptr<ov::op::vX::OpName>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(as_type_ptr<ov::op::vX::OpName>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(as_type_ptr<ov::op::vX::OpName>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(as_type_ptr<ov::op::vX::OpName>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type.get_type_name(),
                       " in evaluate_node<OpName>");
    }
}
```

## High-Level Rules

- Register the op **only** in the latest opset.
- Do **not** edit older `opsetX_tbl.hpp` files or change existing registrations.
- Do **not** break compatibility of existing ops.
- Align style with existing ops in the codebase.

## Output

- Branch or patch with all created/updated files.
- Proceed to **core_op_testing** skill.

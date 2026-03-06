# Skill: Core Op Implementation

> Source: `.github/skills/add-core-op/SKILL.md` (Step 2)
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

## High-Level Rules

- Register the op **only** in the latest opset.
- Do **not** edit older `opsetX_tbl.hpp` files or change existing registrations.
- Do **not** break compatibility of existing ops.
- Align style with existing ops in the codebase.

## Output

- Branch or patch with all created/updated files.
- Proceed to **core_op_testing** skill.

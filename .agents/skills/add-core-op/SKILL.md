---
name: add-core-op
description: Adds a core operator to the OpenVINO toolkit. Use when asked to implement a new operation into OpenVINO.
---

## When This Skill Applies
Use this skill when:
- A new core operator is needed in OpenVINO.
- Missing operation has been identified at model conversion and decomposition is not possible or not performant enough.
- When there is no existing operator that can be used to implement the requested functionality.

## Typical Workflow for Adding Support for a New Op

1. Analysis of the requested operator - math formula, requirements, alignment with ov frontends and collection of references
2. Update of needed files
3. Create tests
4. Create specification describing new op

For more details about executing each step, refer to the sections below.

# Skill Instructions

# Adding a New Operator to OpenVINO
#### Create Header and Source Files
The implementation of a class representing an operator must have corresponding `.hpp` and `.cpp` files. These files should be created in the following locations:
- `**/openvino/src/core/include/openvino/op/new_op_name.hpp`
- `**/openvino/src/core/src/op/new_op_name.cpp`

#### Define the Operator Class
Create a new operator class that is part of `OPENVINO_API` and inherits from the `Op` class:
```cpp
class OPENVINO_API OpName : public Op {
};
```

#### Register the Operator
In the created `.hpp` file, add the following macro adjusted for new op name and opset number:
```cpp
OPENVINO_OP("OpName", "opsetX")
```

#### Implement Constructors
Implement constructor(s) that take inputs and attributes as arguments.
*   An **input** represents a tensor of data or the output of another operator (`Output<Node>`).
*   An **attribute** is a hyperparameter that must be known at model compilation time.
Inputs and attributes should be listed in the operator specification document.

#### Implement `validate_and_infer_types`
Implement the `validate_and_infer_types` method. This method is responsible for:
*   Validating input shapes and types, if restrictions exist
*   Setting the output shapes and element types (precision)

At this stage, input values are usually unknown unless the input is a `Constant`.  
If an input is a `Constant` or can be evaluated, its value may be retrieved. For example, the `axis` input for a Reduce operator is typically provided as a `Constant`, allowing its value to be validated and used for output shape inference.
Input and output shapes may be static or dynamic. The `PartialShape` class is used to represent both cases.

##### Shape Inference
For operators without existing common shape inference function, the shape-related logic should be implemented in a separate `shape_infer` function so it can be shared with plugins. This function should be added to: `**/openvino/src/core/shape_inference/include/`
Following the file name convention, example:
*   `**/openvino/src/core/shape_inference/include/new_op_shape_inference.hpp`

#### Implement `visit_attributes`
Implement the `visit_attributes` method, which is used, for example, by the serialization mechanism.
Each attribute (usually also a class member) must be visited by calling `on_attribute` with the attribute name and value, for example:
```cpp
visitor.on_attribute("axis", m_axis);
```

#### Implement `clone_with_new_inputs`
Implement the `clone_with_new_inputs` method. This method should return a clone of the operator that:
*   Uses the new inputs provided in the input vector
*   Preserves the existing attribute values

#### Conditional Compilation Support
To support conditional compilation, add the following macro at the beginning of every method of the new operator class in the `.cpp` file:
```cpp
OV_OP_SCOPE(<operator_version>_<operator_name>_<method_name>);
```

## High-level rules:
- Register the op only in the latest opset
- Use the latest opset number version for new ops namespace.
- Do not edit older `opsetX_tbl.hpp` files or change existing registrations.
- Don't break compatibility of existing ops

## Files to create/update
Note: Treat it as a double-check; no need to create a file if adding a new version of the op.

Create:
- `**/openvino/src/core/include/openvino/op/new_op_name.hpp` – class declaration.
- `**/openvino/src/core/src/op/new_op_name.cpp` – implementation.
- `**/openvino/src/core/reference/include/openvino/reference/new_op_name.hpp` – reference kernel.

Update :
- `**/openvino/src/core/dev_api/openvino/op/ops_decl.hpp` – register new Op class
- `**/openvino/src/core/include/openvino/op/ops.hpp` – include new op header
- `**/openvino/src/core/include/openvino/opsets/opsetX_tbl.hpp`- register new Op class

## Core Op Class Pattern
Note: Treat it as example, prefer alignment with the code base (see existing ops for style alignment).

**Header (`<op_name>.hpp`):**
- Namespace: `ov::op::vX` (use the latest opset version number for X).
- Base: use an existing utility base when possible, e.g., `util::UnaryElementwiseArithmetic` for unary elementwise ops.
- Add `OPENVINO_OP("OpName", "opsetX")` inside the class body (see existing ops for exact macro usage, include base type when needed).
- Declare:
  - Default constructor.
  - Constructor from `const Output<Node>&` (and additional inputs/attributes as needed).
  - `clone_with_new_inputs`, `evaluate`, `has_evaluate`.

**Source (`<op_name>.cpp`):**
- Include `openvino/op/<op_name>.hpp`, `element_visitor.hpp`, `itt.hpp`, and the reference header if one exists.
- Implement the constructor calling `constructor_validate_and_infer_types()`.
- Implement `clone_with_new_inputs` using `check_new_args_count` and `std::make_shared<OpName>`.
- Implement `evaluate` via `IF_TYPE_OF_CONVERT_TENSORS` that calls `ov::reference::<op_name>`.
- Implement `has_evaluate` to return `true` for supported element types.
- Add `OV_OP_SCOPE(v<X>_<OpName>_<method>)` at the start of each method (constructor body, clone, evaluate, has_evaluate).

## Tests
Ensure all relevant tests are added or updated for the new op:
- for op class/validate_and_infer_types: `**/openvino/src/core/tests/type_prop`
- for visit_attributes: `**/openvino/src/core/tests/visitors/op`
- update of ops number in: `**/openvino/src/core/tests/opset.cpp`
- conformance tests: `**/openvino/src/tests/functional/plugin/conformance/test_runner/op_conformance_runner/src/op_impl_check/single_op_graph.cpp`
- functional op_reference tests: `**/openvino/src/plugins/template/tests/functional/op_reference/`

## Adding Specification
For the newly added operation, a specification must be created as a .rst file following the conventions for other operators.
Example: `**/openvino/docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/signals/istft-16.rst`

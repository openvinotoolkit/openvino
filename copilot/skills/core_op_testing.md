# Skill: Core Op Testing

> Source: `.github/skills/add-core-op/SKILL.md` (Step 3)
> Agent: `core_opspec_agent`

## Prerequisites

- Completed **core_op_implementation** skill - op class is implemented,
  files created and registered.

## Test Categories

### 1. Type Propagation Tests (`type_prop`)

**Location:** `openvino/src/core/tests/type_prop/<op_name>.cpp`

Tests for `validate_and_infer_types`:
- Correct output shape inference for valid inputs.
- Correct output element type inference.
- Static and dynamic input shapes.
- Partial shapes (unknown dimensions).
- Invalid input shapes → expect validation error.
- Invalid input types → expect validation error.
- Constant inputs (e.g. axis) → verify value is used.

### 2. Visitor / Serialization Tests

**Location:** `openvino/src/core/tests/visitors/op/<op_name>.cpp`

Tests for `visit_attributes`:
- All attributes are visited (serialized/deserialized correctly).
- Round-trip: create op → serialize → deserialize → compare attributes.

### 3. Opset Count Update

**Location:** `openvino/src/core/tests/opset.cpp`

- Update the expected op count for the target opset.
- The test verifies the total number of ops registered in each opset table.

### 4. Conformance Tests

**Location:** `openvino/src/tests/functional/plugin/conformance/test_runner/op_conformance_runner/src/op_impl_check/single_op_graph.cpp`

- Add a graph-building function for the new op.
- Used by the conformance test infrastructure to verify plugin support.

### 5. Reference Implementation Tests (`op_reference`)

**Location:** `openvino/src/plugins/template/tests/functional/op_reference/<op_name>.cpp`

- Test the `evaluate()` method against known input/output pairs.
- Cover all supported element types.
- Cover edge cases (empty tensors, scalar inputs, large shapes).
- Use the reference kernel from `openvino/reference/<op_name>.hpp`.

## Execution

```bash
# Build with tests enabled
cmake --build build --target <op_name>_test

# Run type_prop tests
./bin/ov_core_unit_tests --gtest_filter="*<OpName>*"

# Run visitor tests
./bin/ov_core_unit_tests --gtest_filter="*visitor*<OpName>*"

# Run reference tests
./bin/ov_template_func_tests --gtest_filter="*<OpName>*"
```

## Output

- All tests pass → proceed to **core_op_specification** skill.
- Test failures → fix implementation, re-run. Report issues to OV Orchestrator
  if the fix requires changes outside core op scope.

# Skill: Core Op Testing

> Source: `skills/add-core-op/SKILL.md` (Step 3)
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
- Interval-bounded dimensions: `PartialShape{Dimension(2,5), Dimension(1,4), 3}` — verify bounds are preserved.
- Fully-dynamic rank: `PartialShape::dynamic()` — output must also be rank-dynamic.
- Dimension symbol propagation: set symbols on input with `set_shape_symbols()`, verify output has same symbols via `get_shape_symbols()`.
- Scalar input: `Shape{}` (rank-0 tensor).
- **Each scenario should be a separate `TEST` — do not bundle multiple shape cases in one test function.**

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

The conformance framework calls `OpImplCheckTest.checkPluginImplementation` for every registered op.
It looks up a model-building factory per op type. **If no entry is found, the model is `nullptr`
and the test fails with "Target model is empty!".**

For unary elementwise ops (those that inherit from `UnaryElementwiseArithmetic`), add one `else if`
branch to the `generateUnaryEltwise()` function — alphabetically by op name:

```cpp
} else if (ov::is_type<ov::op::v0::Erf>(node)) {
    eltwiseNode = std::make_shared<ov::op::v0::Erf>(param);
} else if (ov::is_type<ov::op::v16::ErfInv>(node)) {   // ← add this
    eltwiseNode = std::make_shared<ov::op::v16::ErfInv>(param);
} else if (ov::is_type<ov::op::v0::Exp>(node)) {
```

For ops that are **not** unary eltwise, add a dedicated `generate(const std::shared_ptr<ov::op::vX::OpName>&)` overload instead (see other examples in the same file).

**Build target (requires `ENABLE_TESTS=ON` in cmake):**

```bash
cmake --preset RelWithDebInfo -DENABLE_TESTS=ON
cmake --build build/RelWithDebInfo --target ov_op_conformance_tests -j$(nproc)
```

### 5. Reference Implementation Tests (`op_reference`)

**Location:** `openvino/src/plugins/template/tests/functional/op_reference/<op_name>.cpp`

- Test the `evaluate()` method against known input/output pairs.
- Cover all supported element types.
- Cover edge cases (empty tensors, scalar inputs, large shapes).
- Use the reference kernel from `openvino/reference/<op_name>.hpp`.
- For domain-restricted ops (e.g. erfinv on (-1,1)): add boundary cases (e.g. x=±1 → ±inf) AND out-of-domain cases (e.g. |x|>1 → NaN).

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

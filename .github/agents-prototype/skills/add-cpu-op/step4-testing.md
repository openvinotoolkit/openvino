# Skill: CPU Op Testing

> Agent: `cpu_agent` — Step 4 of 4

## Prerequisites

- Completed **cpu_op_implementation** — node compiles and has basic execution.
- Completed **cpu_op_optimization** (or skipped if reference-only).

## Test Categories

### 1. Shared Single-Layer Tests (Mandatory)

**Directory:** `src/plugins/intel_cpu/tests/functional/shared_tests_instances/single_layer_tests/`

These tests instantiate the shared OpenVINO layer tests that are common across
all plugins (CPU, GPU, Template). They ensure the CPU implementation produces
results consistent with the reference.

Dynamic shape scenarios can also be tested via shared single-layer tests by
using `ov::test::static_shapes_to_test_representation` with dynamic `InputShape`
entries (e.g., `{{-1, -1}, {{1, 16}, {2, 32}}}`), not only static shapes.

**File to create:** `<op_name>.cpp`

**Pattern:**

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/<op_name>.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::OpNameLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::bf16,
    ov::element::f16,
};

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{1, 16}},
    {{2, 32}},
    {{1, 3, 224, 224}},
    {{2, 64, 7, 7}},
};

// Dynamic shapes can also be tested in shared tests using InputShape:
// const std::vector<std::vector<InputShape>> input_shapes_dynamic = {
//     {{{-1, -1}, {{1, 16}, {2, 32}}}},
//     {{{{1, 4}, {16, 64}}, {{1, 16}, {4, 64}}}},
// };

const auto params = ::testing::Combine(
    ::testing::ValuesIn(
        ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(model_types),
    // Op-specific parameters
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_OpName,
                         OpNameLayerTest,
                         params,
                         OpNameLayerTest::getTestCaseName);

}  // namespace
```

**Test naming:** Use the `smoke_` prefix for basic functionality tests,
`nightly_` for exhaustive tests.

### 2. Custom CPU Single-Layer Tests (Recommended)

**Directory:** `src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/`

These tests verify CPU-specific implementation details that shared tests don't
cover:
- Specific memory layouts (channels-last).
- ISA-specific implementation selection (`jit_avx2`, `jit_avx512`).
- Dynamic shape scenarios.
- Edge cases specific to the CPU implementation.

**File to create:** `<op_name>.cpp`

**Pattern:**

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "openvino/op/<op_name>.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using OpNameLayerCPUTestParamSet = std::tuple<
    InputShape,       // Input shape (supports dynamic)
    ElementType,      // Input element type
    // Op-specific attributes
    CPUSpecificParams  // CPU layout + implementation check
>;

class OpNameLayerCPUTest
    : public testing::WithParamInterface<OpNameLayerCPUTestParamSet>,
      virtual public ov::test::SubgraphBaseTest,
      public CPUTestsBase {
public:
    static std::string getTestCaseName(
        const testing::TestParamInfo<OpNameLayerCPUTestParamSet>& obj) {
        const auto& [shapes, inType, /* attrs, */ cpuParams] = obj.param;
        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "Prc=" << inType << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);
        return results.str();
    }

protected:
    void SetUp() override {
        const auto& [shapes, _inType, /* attrs, */ cpuParams] = this->GetParam();
        inType = _inType;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType = makeSelectedTypeStr(selectedType, inType);
        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(
                std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        // Build the op graph
        auto op = std::make_shared<ov::op::vX::OpName>(
            params[0] /*, attributes */);
        function = create_ov_model(inType, params, op, "OpName");
    }
};

TEST_P(OpNameLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "OpName");
}

namespace {

// ═══════════════════════════════════════════════════════════════════
// CPU-Specific Parameters
// ═══════════════════════════════════════════════════════════════════
const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, {}};
const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, {}};

// ═══════════════════════════════════════════════════════════════════
// Static Shape Tests
// ═══════════════════════════════════════════════════════════════════
const std::vector<InputShape> staticShapes4D = {
    {{}, {{1, 16, 8, 8}}},
    {{}, {{2, 32, 4, 4}}},
    {{}, {{1, 64, 1, 1}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_OpName_Static,
    OpNameLayerCPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(staticShapes4D),
        ::testing::Values(ElementType::f32, ElementType::bf16),
        // Op-specific attribute values
        ::testing::Values(cpuParams_nchw, cpuParams_nhwc)),
    OpNameLayerCPUTest::getTestCaseName);

// ═══════════════════════════════════════════════════════════════════
// Dynamic Shape Tests
// ═══════════════════════════════════════════════════════════════════
const std::vector<InputShape> dynamicShapes4D = {
    // {PartialShape, list of concrete shapes to test}
    {{-1, -1, -1, -1}, {{1, 16, 8, 8}, {2, 32, 4, 4}, {1, 64, 1, 1}}},
    {{{1, 4}, {16, 64}, {1, 16}, {1, 16}}, {{1, 16, 8, 8}, {4, 64, 16, 16}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_OpName_Dynamic,
    OpNameLayerCPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(dynamicShapes4D),
        ::testing::Values(ElementType::f32),
        // Op-specific attribute values
        ::testing::Values(cpuParams_nchw)),
    OpNameLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
```

### Eltwise-Routed Ops: Use Activation Test Infrastructure

For ops routed through the existing `Eltwise` node (see **cpu_op_implementation** Fast Path),
**do not create a separate custom test file**. Instead, extend the existing activation test infrastructure:

**`src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/classes/activation.cpp`:**
1. Add `{ActivationTypes::OpName, {{}}}` to `activationTypes()` (or `activationTypesDynamicMath()` for ref-only ops).
2. Add `OpName` to `getPrimitiveType()` jit lists for ARM64 and riscv64.
3. Add edge-case injection in `ActivationLayerCPUTest::generate_inputs()` for domain-bounded ops.

**`src/plugins/intel_cpu/tests/functional/shared_tests_instances/single_layer_tests/activation.cpp`:**
- The op is already covered by the parameterized `ActivationLayerTest` and `ActivationLayerCPUTest` once added to the `activationTypes()` map — no new `INSTANTIATE_TEST_SUITE_P` needed.

### 3. Edge Cases to Cover

| Category | Test Cases |
|----------|-----------|
| **Shapes** | Scalar input, 1-element tensor, very large tensor, zero-dimension (empty) |
| **Precisions** | `f32`, `bf16`, `f16`, `i8`, `u8`, `i32`, `i64` (as applicable) |
| **Layouts** | Planar (`ncsp`), channels-last (`nspc`) |
| **Dynamic shapes** | Fully dynamic (`{-1, -1, ...}`), partially dynamic, shape-changing across inferences |
| **Rank variation** | 1D, 2D, 3D, 4D, 5D (as supported by the op) |
| **Attribute values** | All valid attribute combinations, especially boundary values |
| **Numerical stability** | Very small values (denormals), very large values, NaN, Inf |

### 4. Verifying ISA Implementation Selection

Custom CPU tests can verify that the correct implementation type is selected:

```cpp
TEST_P(OpNameLayerCPUTest, CompareWithRefs) {
    run();
    // Verify the correct CPU implementation was chosen
    CheckPluginRelatedResults(compiledModel, "OpName");
}
```

The `CheckPluginRelatedResults` function validates that `selectedType`
(e.g., `jit_avx512_FP32`) matches the expected implementation based on the
current hardware.

## Build and Run Tests

### Build

```bash
cd build

# Build shared functional tests
cmake --build . --target ov_cpu_func_tests -j$(nproc)

# Or build only the CPU plugin library
cmake --build . --target openvino_intel_cpu_plugin -j$(nproc)
```

### Run Shared Tests

```bash
./bin/intel64/Release/ov_cpu_func_tests \
    --gtest_filter=*smoke*OpName*
```

### Run Custom CPU Tests

```bash
./bin/intel64/Release/ov_cpu_func_tests \
    --gtest_filter=*OpNameLayerCPUTest*
```

### Run All Tests for the Op

```bash
./bin/intel64/Release/ov_cpu_func_tests \
    --gtest_filter=*OpName*
```

### Run with Verbose Output

```bash
./bin/intel64/Release/ov_cpu_func_tests \
    --gtest_filter=*OpName* \
    --gtest_print_time=1
```

## Debug Capabilities

The CPU plugin provides debug features documented in
`src/plugins/intel_cpu/docs/debug_capabilities/`:

```bash
# Print execution graph with performance counters
export OV_CPU_DEBUG_LOG=1

# Dump node execution details
export OV_CPU_EXEC_GRAPH_INFO=1
```

## Test Failure Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Test not found | Test not registered in CMake or wrong filter | Check test file is under correct directory |
| Accuracy mismatch | Incorrect precision handling | Verify data type dispatch and precision conversion |
| Segfault | Memory stride/offset calculation error | Check `getStrides()`, `getStaticDims()` usage |
| Shape mismatch | Incorrect shape inference | Verify `NgraphShapeInferFactory` or custom shape inference |
| Wrong impl selected | `initSupportedPrimitiveDescriptors` ordering | First matching descriptor is selected — order matters |
| Dynamic shape failure | Missing `executeDynamicImpl` | Ensure `executeDynamicImpl` delegates to `execute` |

## Output

- All shared single-layer tests pass.
- Custom CPU single-layer tests pass (static + dynamic shapes).
- Report `success` + list of files created to OV Orchestrator.

### Files Created in This Step

| File | Purpose |
|------|---------|
| `src/plugins/intel_cpu/tests/functional/shared_tests_instances/single_layer_tests/<op_name>.cpp` | Shared test instantiation |
| `src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/<op_name>.cpp` | Custom CPU-specific tests |

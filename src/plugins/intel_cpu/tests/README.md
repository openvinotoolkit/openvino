# CPU plugin functional tests

How CPU plugin functional tests are organized and written. This is general
reference for anyone adding or modifying CPU tests; for the workflow of adding a
*new* op (which files to create) see
[docs/op_development](../docs/op_development/README.md).

## Contents

- [Directory structure](#directory-structure)
- [Test categories](#test-categories)
- [Shared single-layer tests](#shared-single-layer-tests)
- [Custom CPU single-layer tests](#custom-cpu-single-layer-tests)
- [Eltwise-routed ops: the activation test infrastructure](#eltwise-routed-ops-the-activation-test-infrastructure)
- [Verifying implementation selection](#verifying-implementation-selection)
- [Edge cases to cover](#edge-cases-to-cover)
- [Troubleshooting](#troubleshooting)

## Directory structure

Taken CPU plugin specific single layer tests as an example:

``` shell
single_layer_tests/
├── classes # test classes
│   ├── activation.cpp # test class with common parameters
│   ├── activation.h
│   ├── convolution.cpp
│   └── convolution.h
└── instances # test instances
    ├── arm # arch specific instances if any
    │   ├── acl # backend specific instances if any
    │   │   └── actication.cpp 
    │   └── onednn
    │       └── convolution.cpp
    ├── common # common instances across all the architecture
    │   ├── activation.cpp
    │   └── onednn
    │       └── convolution.cpp
    ├── _some_new_arch
    │   ├── activation.cpp
    │   └── convolution.cpp
    └── x64 # arch specific instances if any
        ├── activation.cpp # native instances for the arch (no backend involved)
        ├── onednn # backend specific instances if any
        │   └── convolution.cpp
        └── _some_new_backend
            └── convolution.cpp
```

## Test categories

A CPU operation is exercised by two complementary kinds of single-layer test:

1. **Shared single-layer tests** (mandatory) — instantiate the OpenVINO layer
   tests common to all plugins (CPU, GPU, Template), ensuring CPU results match the
   reference. Located under
   `functional/shared_tests_instances/single_layer_tests/`.
2. **Custom CPU single-layer tests** (recommended) — verify CPU-specific details
   the shared tests don't cover: memory layouts, ISA-specific implementation
   selection, dynamic shapes, and edge cases. Located under
   `functional/custom/single_layer_tests/`.

The shared CPU test base classes (`CPUTestsBase`, `CPUSpecificParams`,
`CheckPluginRelatedResults`, `makeSelectedTypeStr`) live in
[functional/utils/cpu_test_utils.hpp](functional/utils/cpu_test_utils.hpp);
fusing helpers in [functional/utils/fusing_test_utils.hpp](functional/utils/fusing_test_utils.hpp).

Use the `smoke_` prefix for basic functionality tests and `nightly_` for
exhaustive ones.

## Shared single-layer tests

Create `<op_name>.cpp` under
`functional/shared_tests_instances/single_layer_tests/`. Dynamic shapes can be
tested here too via `ov::test::static_shapes_to_test_representation` with dynamic
`InputShape` entries (e.g. `{{-1, -1}, {{1, 16}, {2, 32}}}`), not only static
shapes.

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

// Dynamic shapes can also be tested using InputShape:
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

## Custom CPU single-layer tests

Create `<op_name>.cpp` under `functional/custom/single_layer_tests/`. The test
class combines `WithParamInterface`, `SubgraphBaseTest`, and `CPUTestsBase`, and
validates the selected implementation in the test body.

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

const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, {}};
const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, {}};

// Static shapes
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

// Dynamic shapes: {PartialShape, list of concrete shapes to test}
const std::vector<InputShape> dynamicShapes4D = {
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

## Eltwise-routed ops: the activation test infrastructure

Ops routed through the existing `Eltwise` node do **not** get a separate custom
test file. Instead, extend the activation test infrastructure:

**`functional/custom/single_layer_tests/classes/activation.cpp`:**
1. Add `{ActivationTypes::OpName, {{}}}` to `activationTypes()` (or
   `activationTypesDynamicMath()` for ref-only ops).
2. Add `OpName` to `getPrimitiveType()` jit lists for ARM64 and riscv64.
3. Add edge-case injection in `ActivationLayerCPUTest::generate_inputs()` for
   domain-bounded ops.

**`functional/shared_tests_instances/single_layer_tests/activation.cpp`:** once the
op is in the `activationTypes()` map it is covered by the parameterized
`ActivationLayerTest` / `ActivationLayerCPUTest` — no new `INSTANTIATE_TEST_SUITE_P`.

## Verifying implementation selection

`CheckPluginRelatedResults(compiledModel, "OpName")` in the test body validates
that the chosen `selectedType` (e.g. `jit_avx512_FP32`) matches the implementation
expected on the current hardware — this is how a custom CPU test confirms the
right ISA-specific path was selected. `selectedType` is derived in `SetUp()` from
`CPUSpecificParams` (or `getPrimitiveType()` when left empty).

## Edge cases to cover

| Category | Test Cases |
|----------|-----------|
| **Shapes** | Scalar input, 1-element tensor, very large tensor, zero-dimension (empty) |
| **Precisions** | `f32`, `bf16`, `f16`, `i8`, `u8`, `i32`, `i64` (as applicable) |
| **Layouts** | Planar (`ncsp`), channels-last (`nspc`) |
| **Dynamic shapes** | Fully dynamic (`{-1, -1, ...}`), partially dynamic, shape-changing across inferences |
| **Rank variation** | 1D, 2D, 3D, 4D, 5D (as supported by the op) |
| **Attribute values** | All valid attribute combinations, especially boundary values |
| **Numerical stability** | Very small values (denormals), very large values, NaN, Inf |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Test not found | Test not registered in CMake or wrong filter | Check test file is under correct directory |
| Accuracy mismatch | Incorrect precision handling | Verify data type dispatch and precision conversion |
| Segfault | Memory stride/offset calculation error | Check `getStrides()`, `getStaticDims()` usage |
| Shape mismatch | Incorrect shape inference | Verify `NgraphShapeInferFactory` or custom shape inference |
| Wrong impl selected | `initSupportedPrimitiveDescriptors` ordering | First matching descriptor is selected — order matters |
| Dynamic shape failure | Missing `executeDynamicImpl` | Ensure `executeDynamicImpl` delegates to `execute` |

For deeper diagnosis (verbose execution tracing, blob dumping, execution-graph
serialization, per-node inference precision, logging), use the plugin's debug
capabilities — see [debug capabilities](../docs/debug_capabilities/README.md).

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <set>

#include "custom/single_layer_tests/classes/activation.hpp"
#if defined(OPENVINO_ARCH_ARM64)
#    include "custom/single_layer_tests/classes/comparison.hpp"
#endif
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Activation {

/* ============= Activation (1D) ============= */
const auto basicCases3D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic3D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams3D())),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation3D_Eltwise_CPU, ActivationLayerCPUTest, basicCases3D, ActivationLayerCPUTest::getTestCaseName);

const auto basicCasesSnippets3D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic3D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesSnippets())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams3D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation3D_Snippets_CPU, ActivationLayerCPUTest, basicCasesSnippets3D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (2D) ============= */
const auto basicCases4D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic4D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D())),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation4D_Eltwise_CPU, ActivationLayerCPUTest, basicCases4D, ActivationLayerCPUTest::getTestCaseName);

const auto basicCasesSnippets4D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic4D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesSnippets())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation4D_Snippets_CPU, ActivationLayerCPUTest, basicCasesSnippets4D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (5D) ============= */
const auto basicCases5D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic5D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5D())),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_Eltwise_CPU, ActivationLayerCPUTest, basicCases5D, ActivationLayerCPUTest::getTestCaseName);

const auto basicCasesSnippets5D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic5D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesSnippets())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_Snippets_CPU, ActivationLayerCPUTest, basicCasesSnippets5D, ActivationLayerCPUTest::getTestCaseName);

const auto dynamicMathBasicCases = ::testing::Combine(
    ::testing::ValuesIn(dynamicMathBasic()),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesDynamicMath())),
    ::testing::ValuesIn(netPrecisions()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(cpuParamsDynamicMath()),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_dynamicMath_CPU, ActivationLayerCPUTest, dynamicMathBasicCases, ActivationLayerCPUTest::getTestCaseName);

#if defined(OPENVINO_ARCH_ARM64)
namespace {

class ActivationLayerI32CPUTest : public ActivationLayerCPUTest {
protected:
    void SetUp() override {
        ActivationLayerCPUTest::SetUp();
        abs_threshold = 0;
        rel_threshold = 0;
        selectedType = "jit_i32";
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        ActivationLayerCPUTest::generate_inputs(targetInputStaticShapes);
        const auto activation = std::get<2>(this->GetParam()).first;
        const size_t windowIndex = activation == utils::ActivationTypes::Floor ? 0 : 1;
        const auto& window = *comparison::i32Windows().at(windowIndex);
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            inputs[funcInput.get_node_shared_ptr()] = ov::test::utils::create_and_fill_tensor(
                funcInput.get_element_type(),
                targetInputStaticShapes[i],
                utils::InputGenerateData(window.start_from, window.range, 1, static_cast<int32_t>(i + 1)));
        }
    }
};

TEST_P(ActivationLayerI32CPUTest, CompareWithRefs) {
    run();
    if (std::get<7>(this->GetParam())) {
        CheckNumberOfNodesWithType(compiledModel, "Subgraph", 1);
    }
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

// Scalar and full-vector execution are covered by the existing suites; the exact i32 cases use a tail and dynamic input.
const std::vector<std::vector<InputShape>> i32ActivationShapes = {
    {{{}, {{1, 29}}}},
    {{{-1, -1}, {{1, 29}, {3, 4}, {5, 1}}}},
};

const auto i32ActivationParams = ::testing::Combine(
    ::testing::ValuesIn(i32ActivationShapes),
    ::testing::Values(activationShapes()),
    ::testing::Values(std::make_pair(utils::ActivationTypes::Abs, std::vector<float>{}),
                      std::make_pair(utils::ActivationTypes::Negative, std::vector<float>{}),
                      std::make_pair(utils::ActivationTypes::Floor, std::vector<float>{})),
    ::testing::Values(ov::element::i32),
    ::testing::Values(ov::element::i32),
    ::testing::Values(ov::element::i32),
    ::testing::Values(emptyCPUSpec),
    ::testing::Values(false, true));

INSTANTIATE_TEST_SUITE_P(smoke_Activation_I32Exact_ARM64,
                         ActivationLayerI32CPUTest,
                         i32ActivationParams,
                         ActivationLayerI32CPUTest::getTestCaseName);

}  // namespace
#endif

} // namespace Activation
}  // namespace test
}  // namespace ov

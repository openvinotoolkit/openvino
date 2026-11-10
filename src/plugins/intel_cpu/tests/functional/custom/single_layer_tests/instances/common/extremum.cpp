// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <set>

#include "custom/single_layer_tests/classes/extremum.hpp"
#if defined(OPENVINO_ARCH_ARM64)
#    include "custom/single_layer_tests/classes/comparison.hpp"
#endif
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace extremum {

const auto basicCasesSnippets = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(inputShape())),
    ::testing::ValuesIn(extremumTypes()),
    ::testing::ValuesIn(netPrecisions()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Extremum_Snippets_CPU, ExtremumLayerCPUTest, basicCasesSnippets, ExtremumLayerCPUTest::getTestCaseName);

#if defined(OPENVINO_ARCH_ARM64)
namespace {

class ExtremumLayerI32CPUTest : public ExtremumLayerCPUTest {
protected:
    void SetUp() override {
        ExtremumLayerCPUTest::SetUp();
        abs_threshold = 0;
        rel_threshold = 0;
        selectedType = "jit_i32";
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        const auto extremum = std::get<1>(this->GetParam());
        const size_t windowIndex = extremum == utils::MinMaxOpType::MINIMUM ? 2 : 3;
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

TEST_P(ExtremumLayerI32CPUTest, CompareWithRefs) {
    run();
    if (std::get<6>(this->GetParam())) {
        CheckNumberOfNodesWithType(compiledModel, "Subgraph", 1);
    }
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

// Scalar, vector and dynamic execution are covered by the existing suites; use a tail and a broadcast here.
const std::vector<std::vector<InputShape>> i32ExtremumShapes = {
    {{{}, {{1, 29}}}, {{}, {{1, 29}}}},
    {{{}, {{2, 29}}}, {{}, {{1, 1}}}},
};

const auto i32ExtremumParams = ::testing::Combine(::testing::ValuesIn(i32ExtremumShapes),
                                                  ::testing::ValuesIn(extremumTypes()),
                                                  ::testing::Values(ov::element::i32),
                                                  ::testing::Values(ov::element::i32),
                                                  ::testing::Values(ov::element::i32),
                                                  ::testing::Values(emptyCPUSpec),
                                                  ::testing::Values(false, true));

INSTANTIATE_TEST_SUITE_P(smoke_Extremum_I32Exact_ARM64,
                         ExtremumLayerI32CPUTest,
                         i32ExtremumParams,
                         ExtremumLayerI32CPUTest::getTestCaseName);

}  // namespace
#endif

}  // namespace extremum
}  // namespace test
}  // namespace ov

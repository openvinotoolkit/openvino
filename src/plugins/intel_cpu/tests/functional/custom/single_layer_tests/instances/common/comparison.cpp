// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/comparison.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace comparison {

const auto withParameterCases = ::testing::Combine(::testing::ValuesIn(inShapesWithParameter()),
                                                   ::testing::ValuesIn(comparisonTypes()),
                                                   ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                                   ::testing::ValuesIn(modelPrc()),
                                                   ::testing::ValuesIn(inferPrc()),
                                                   ::testing::ValuesIn(enforceSnippets()),
                                                   ::testing::Values(false),
                                                   ::testing::Values(ComparisonGraphType::COMPARE));
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_WithParameter, ComparisonLayerCPUTest, withParameterCases, ComparisonLayerCPUTest::getTestCaseName);

const auto withConstantCases = ::testing::Combine(::testing::ValuesIn(inShapesWithConstant()),
                                                  ::testing::ValuesIn(comparisonTypes()),
                                                  ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                                  ::testing::ValuesIn(modelPrc()),
                                                  ::testing::ValuesIn(inferPrc()),
                                                  ::testing::ValuesIn(enforceSnippets()),
                                                  ::testing::Values(false),
                                                  ::testing::Values(ComparisonGraphType::COMPARE));
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_WithConstant, ComparisonLayerCPUTest, withConstantCases, ComparisonLayerCPUTest::getTestCaseName);

#if defined(OPENVINO_ARCH_ARM64)
const std::vector<std::vector<InputShape>> i32PrecisionShapes = {
    {{{}, {{1, 1}}}, {{}, {{1, 1}}}},
    {{{}, {{1, 29}}}, {{}, {{1, 29}}}},
    {{{}, {{1, 35}}}, {{}, {{1, 35}}}},
    {{{}, {{1, 37}}}, {{}, {{1, 37}}}},
    {{{}, {{2, 29}}}, {{}, {{1, 1}}}},
    {{{-1, -1}, {{2, 29}, {5, 1}, {1, 1}, {3, 29}}}, {{-1, -1}, {{1, 29}, {1, 1}, {1, 1}, {1, 29}}}},
};

const auto i32PrecisionCases = ::testing::Combine(
    ::testing::ValuesIn(i32PrecisionShapes),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT),
    ::testing::Values(ov::element::i32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(enforceSnippets()),
    ::testing::Values(true),
    ::testing::Values(ComparisonGraphType::COMPARE, ComparisonGraphType::COMPARE_LOGICAL_NOT));

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_I32Precision,
                         ComparisonLayerCPUTest,
                         i32PrecisionCases,
                         ComparisonLayerCPUTest::getTestCaseName);

const auto i32PrecisionAddCases = ::testing::Combine(
    ::testing::ValuesIn(i32PrecisionShapes),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT),
    ::testing::Values(ov::element::i32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(enforceSnippets()),
    ::testing::Values(true),
    ::testing::Values(ComparisonGraphType::ADD_COMPARE));

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_I32Precision_Add,
                         ComparisonLayerCPUTest,
                         i32PrecisionAddCases,
                         ComparisonLayerCPUTest::getTestCaseName);
#endif

}  // namespace comparison
}  // namespace test
}  // namespace ov

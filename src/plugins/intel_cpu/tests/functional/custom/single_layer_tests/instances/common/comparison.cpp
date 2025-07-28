// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/comparison.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace comparison {

const auto unaryEltwiseWithParameterCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithParameter()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
    ::testing::ValuesIn(modelPrc()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::Values(false)
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_Eltwise_WithParameter_CPU, ComparisonLayerCPUTest, unaryEltwiseWithParameterCases, ComparisonLayerCPUTest::getTestCaseName);

const auto unaryEltwiseWithConstantCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithConstant()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
    ::testing::ValuesIn(modelPrc()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::Values(false)
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_Eltwise_WithConstant_CPU, ComparisonLayerCPUTest, unaryEltwiseWithConstantCases, ComparisonLayerCPUTest::getTestCaseName);

const auto unarySnippetsWithParameterCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithParameter()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
    ::testing::ValuesIn(modelPrc()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::Values(true)
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_Snippets_WithParameter_CPU, ComparisonLayerCPUTest, unarySnippetsWithParameterCases, ComparisonLayerCPUTest::getTestCaseName);

const auto unarySnippetsWithConstantCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithConstant()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
    ::testing::ValuesIn(modelPrc()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::Values(true)
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_Snippets_WithConstant_CPU, ComparisonLayerCPUTest, unarySnippetsWithConstantCases, ComparisonLayerCPUTest::getTestCaseName);

}  // namespace comparison
}  // namespace test
}  // namespace ov

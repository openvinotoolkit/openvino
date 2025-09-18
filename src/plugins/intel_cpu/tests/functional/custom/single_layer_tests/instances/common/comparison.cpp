// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/comparison.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace comparison {

const auto withParameterCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithParameter()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
    ::testing::ValuesIn(modelPrc()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::ValuesIn(enforceSnippets())
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_WithParameter, ComparisonLayerCPUTest, withParameterCases, ComparisonLayerCPUTest::getTestCaseName);

const auto withConstantCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithConstant()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
    ::testing::ValuesIn(modelPrc()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::ValuesIn(enforceSnippets())
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_WithConstant, ComparisonLayerCPUTest, withConstantCases, ComparisonLayerCPUTest::getTestCaseName);

}  // namespace comparison
}  // namespace test
}  // namespace ov

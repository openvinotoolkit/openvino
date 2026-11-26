// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/comparison.hpp"

#include <cstdint>
#include <limits>

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
    ::testing::ValuesIn(enforceSnippets()),
    ::testing::Values(std::nullopt)
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_WithParameter, ComparisonLayerCPUTest, withParameterCases, ComparisonLayerCPUTest::getTestCaseName);

const auto withConstantCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithConstant()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
    ::testing::ValuesIn(modelPrc()),
    ::testing::ValuesIn(inferPrc()),
    ::testing::ValuesIn(enforceSnippets()),
    ::testing::Values(std::nullopt)
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_WithConstant, ComparisonLayerCPUTest, withConstantCases, ComparisonLayerCPUTest::getTestCaseName);

#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_X86_64)
namespace {

const auto i32WithParameterCases = ::testing::Combine(::testing::ValuesIn(inShapesWithParameter()),
                                                      ::testing::ValuesIn(comparisonTypes()),
                                                      ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                                      ::testing::Values(ov::element::i32),
                                                      ::testing::Values(ov::element::f32),
                                                      ::testing::ValuesIn(enforceSnippets()),
                                                      ::testing::ValuesIn(i32Windows()));
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_I32Exact_WithParameter,
                         ComparisonLayerCPUTest,
                         i32WithParameterCases,
                         ComparisonLayerCPUTest::getTestCaseName);

const auto i32WithConstantCases = ::testing::Combine(::testing::ValuesIn(inShapesWithConstant()),
                                                     ::testing::ValuesIn(comparisonTypes()),
                                                     ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                                     ::testing::Values(ov::element::i32),
                                                     ::testing::Values(ov::element::f32),
                                                     ::testing::ValuesIn(enforceSnippets()),
                                                     ::testing::ValuesIn(i32Windows()));
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_I32Exact_WithConstant,
                         ComparisonLayerCPUTest,
                         i32WithConstantCases,
                         ComparisonLayerCPUTest::getTestCaseName);

}  // namespace
#endif

}  // namespace comparison
}  // namespace test
}  // namespace ov

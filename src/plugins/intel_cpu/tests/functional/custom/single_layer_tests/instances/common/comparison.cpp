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

// i32 values that are not exactly representable in f32: around +-2^24 and near the i32 limits. Each window holds 3
// consecutive values, so both inputs are often equal or differ by 1.
const std::vector<std::optional<ov::test::utils::InputGenerateData>>& i32Windows() {
    static const std::vector<std::optional<ov::test::utils::InputGenerateData>> windows{
        ov::test::utils::InputGenerateData((1 << 24) - 1, 3),
        ov::test::utils::InputGenerateData(-(1 << 24) - 1, 3),
        ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::max() - 2, 3),
        ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::min(), 3),
    };

    return windows;
}

// Exercise scalar execution, a complete SIMD vector and a vector tail using the same windows.
std::vector<std::vector<InputShape>> i32PrecisionShapes(std::vector<std::vector<InputShape>> shapes) {
    for (const size_t length : {1, 4, 29}) {
        shapes.push_back({{{}, {{1, length}}}, {{}, {{1, length}}}});
    }
    return shapes;
}

const auto i32WithParameterCases = ::testing::Combine(::testing::ValuesIn(i32PrecisionShapes(inShapesWithParameter())),
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

const auto i32WithConstantCases = ::testing::Combine(::testing::ValuesIn(i32PrecisionShapes(inShapesWithConstant())),
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

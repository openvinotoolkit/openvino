// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <limits>

#include "custom/single_layer_tests/classes/comparison.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace comparison {
namespace {

// i32 values that are not exactly representable in f32: around +-2^24 and near the i32 limits. Each window holds 3
// consecutive values, so both inputs are often equal or differ by 1.
const std::vector<std::optional<ov::test::utils::InputGenerateData>>& i32Windows() {
    static const std::vector<std::optional<ov::test::utils::InputGenerateData>> windows {
        ov::test::utils::InputGenerateData((1 << 24) - 1, 3),
        ov::test::utils::InputGenerateData(-(1 << 24) - 1, 3),
        ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::max() - 2, 3),
        ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::min(), 3),
    };

    return windows;
}

const auto i32WithParameterCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithParameter()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
    ::testing::Values(ov::element::i32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(enforceSnippets()),
    ::testing::ValuesIn(i32Windows())
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_I32Exact_WithParameter, ComparisonLayerCPUTest, i32WithParameterCases, ComparisonLayerCPUTest::getTestCaseName);

const auto i32WithConstantCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesWithConstant()),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
    ::testing::Values(ov::element::i32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(enforceSnippets()),
    ::testing::ValuesIn(i32Windows())
);
INSTANTIATE_TEST_SUITE_P(smoke_Comparison_I32Exact_WithConstant, ComparisonLayerCPUTest, i32WithConstantCases, ComparisonLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace comparison
}  // namespace test
}  // namespace ov

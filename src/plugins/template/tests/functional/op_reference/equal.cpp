// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/equal.hpp"

#include <gtest/gtest.h>

#include "comparison.hpp"

using namespace ov;

namespace reference_tests {
namespace ComparisonOpsRefTestDefinitions {
namespace {

template <element::Type_t IN_ET>
std::vector<RefComparisonParams> generateComparisonParams(const element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<RefComparisonParams> compParams{
        // 1D // 2D // 3D // 4D
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{2, 2}, type, std::vector<T>{0, 12, 23, 0}})
            .input2({{2, 2}, type, std::vector<T>{0, 12, 23, 0}})
            .expected({{2, 2}, element::boolean, std::vector<char>{1, 1, 1, 1}}),
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{2, 3}, type, std::vector<T>{0, 6, 45, 1, 21, 21}})
            .input2({{2, 3}, type, std::vector<T>{1, 18, 23, 1, 19, 21}})
            .expected({{2, 3}, element::boolean, std::vector<char>{0, 0, 0, 1, 0, 1}}),
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{1}, type, std::vector<T>{53}})
            .input2({{1}, type, std::vector<T>{53}})
            .expected({{1}, element::boolean, std::vector<char>{1}}),
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{2, 4}, type, std::vector<T>{0, 12, 23, 0, 1, 5, 11, 8}})
            .input2({{2, 4}, type, std::vector<T>{0, 12, 23, 0, 10, 5, 11, 8}})
            .expected({{2, 4}, element::boolean, std::vector<char>{1, 1, 1, 1, 0, 1, 1, 1}}),
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{3, 1, 2}, type, std::vector<T>{2, 1, 4, 1, 3, 1}})
            .input2({{1, 2, 1}, type, std::vector<T>{1, 1}})
            .expected({{3, 2, 2}, element::boolean, std::vector<char>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}}),
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{2, 1, 2, 1}, type, std::vector<T>{2, 1, 4, 1}})
            .input2({{1, 2, 1}, type, std::vector<T>{1, 1}})
            .expected({{2, 1, 2, 1}, element::boolean, std::vector<char>{0, 1, 0, 1}})};
    return compParams;
}

std::vector<RefComparisonParams> generateComparisonCombinedParams() {
    const std::vector<std::vector<RefComparisonParams>> compTypeParams{
        generateComparisonParams<element::Type_t::f32>(element::f32),
        generateComparisonParams<element::Type_t::f16>(element::f16),
        generateComparisonParams<element::Type_t::i32>(element::i32),
        generateComparisonParams<element::Type_t::u32>(element::u32),
        generateComparisonParams<element::Type_t::u8>(element::boolean)};
    std::vector<RefComparisonParams> combinedParams;

    for (const auto& params : compTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_With_Hardcoded_Refs,
                         ReferenceComparisonLayerTest,
                         ::testing::ValuesIn(generateComparisonCombinedParams()),
                         ReferenceComparisonLayerTest::getTestCaseName);

template <element::Type_t IN_ET>
std::vector<RefComparisonParams> generateNumericParams(const element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<RefComparisonParams> compParams{
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{4}, type, std::vector<T>{-2.5f, 25.5f, 2.25f, NAN}})
            .input2({{4}, type, std::vector<T>{10.0f, 5.0f, 2.25f, 10.0f}})
            .expected({{4},
                       element::boolean,
                       std::vector<char>{
                           0,
                           0,
                           1,
                           0,
                       }}),
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{2, 3}, type, std::vector<T>{0.0f, NAN, NAN, 1.0f, 21.0f, -INFINITY}})
            .input2({{2, 3}, type, std::vector<T>{1.0f, NAN, 23.0f, 1.0f, 19.0f, 21.0f}})
            .expected({{2, 3}, element::boolean, std::vector<char>{0, 0, 0, 1, 0, 0}}),
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{1}, type, std::vector<T>{INFINITY}})
            .input2({{1}, type, std::vector<T>{INFINITY}})
            .expected({{1}, element::boolean, std::vector<char>{1}}),
        Builder{}
            .compType(ComparisonTypes::EQUAL)
            .input1({{5}, type, std::vector<T>{-2.5f, 25.5f, 2.25f, INFINITY, 6.0f}})
            .input2({{5}, type, std::vector<T>{10.0f, 5.0f, 2.25f, 10.0f, -INFINITY}})
            .expected({{5}, element::boolean, std::vector<char>{0, 0, 1, 0, 0}})};
    return compParams;
}

std::vector<RefComparisonParams> generateNumericCombinedParams() {
    const std::vector<std::vector<RefComparisonParams>> compTypeParams{
        generateNumericParams<element::Type_t::f16>(element::f16),
        generateNumericParams<element::Type_t::f32>(element::f32)};
    std::vector<RefComparisonParams> combinedParams;

    for (const auto& params : compTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Numeric_With_Hardcoded_Refs,
                         ReferenceComparisonLayerTest,
                         ::testing::ValuesIn(generateNumericCombinedParams()),
                         ReferenceComparisonLayerTest::getTestCaseName);
}  // namespace
}  // namespace ComparisonOpsRefTestDefinitions
}  // namespace reference_tests

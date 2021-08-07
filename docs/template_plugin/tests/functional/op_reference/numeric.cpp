// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "comparison.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using ComparisonTypes = ngraph::helpers::ComparisonTypes;


namespace reference_tests {
namespace ComparisonOpsRefTestDefinitions {
namespace {

TEST_P(ReferenceComparisonLayerTest, NumericCompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<RefComparisonParams> generateComparisonParams(const element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<RefComparisonParams> compParams {
        Builder {}
            .compType(ComparisonTypes::EQUAL)
            .input1({{4}, type, std::vector<T> {-2.5f, 25.5f, 2.25f, NAN}})
            .input2({{4}, type, std::vector<T> {10.0f, 5.0f, 2.25f, 10.0f}})
            .expected({{4}, element::boolean, std::vector<char> {0, 0, 1, 0, }}),
        Builder {}
            .compType(ComparisonTypes::EQUAL)
            .input1({{2, 3}, type, std::vector<T> {0.0f, NAN, NAN, 1.0f, 21.0f, -INFINITY}})
            .input2({{2, 3}, type, std::vector<T> {1.0f, NAN, 23.0f, 1.0f, 19.0f, 21.0f}})
            .expected({{2, 3}, element::boolean, std::vector<char> {0, 0, 0, 1, 0, 0}}),
        Builder {}
            .compType(ComparisonTypes::EQUAL)
            .input1({{1}, type, std::vector<T> {INFINITY}})
            .input2({{1}, type, std::vector<T> {INFINITY}})
            .expected({{1}, element::boolean, std::vector<char> {1}}),
        Builder {}
            .compType(ComparisonTypes::EQUAL)
            .input1({{5}, type, std::vector<T> {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f}})
            .input2({{5}, type, std::vector<T> {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY}})
            .expected({{5}, element::boolean, std::vector<char> {0, 0, 1, 0, 0}})};
    return compParams;
}

std::vector<RefComparisonParams> generateComparisonCombinedParams() {
    const std::vector<std::vector<RefComparisonParams>> compTypeParams {
        generateComparisonParams<element::Type_t::f16>(element::f16),
        generateComparisonParams<element::Type_t::f32>(element::f32)};
    std::vector<RefComparisonParams> combinedParams;

    for (const auto& params : compTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Numeric_With_Hardcoded_Refs, ReferenceComparisonLayerTest, ::testing::ValuesIn(generateComparisonCombinedParams()),
                         ReferenceComparisonLayerTest::getTestCaseName);
} // namespace
} // namespace ComparisonOpsRefTestDefinitions
} // namespace reference_tests

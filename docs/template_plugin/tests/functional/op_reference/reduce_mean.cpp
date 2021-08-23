// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "reduction.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using ReductionType = ngraph::helpers::ReductionType;

namespace reference_tests {
namespace ReductionOpsRefTestDefinitions {
namespace {

template <element::Type_t IN_ET,
          typename std::enable_if<!std::is_integral<typename element_type_traits<IN_ET>::value_type>::value, bool>::type = true>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {
        ReductionParams(PartialShape{2, 2}, IN_ET, std::vector<T>{1, 2, 3, 4},
                        std::vector<int64_t>{0, 1}, std::vector<T>{2.5}, keep_dims, ReductionType::Mean),
        ReductionParams(PartialShape{3, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6},
                        std::vector<int64_t>{0}, std::vector<T>{3, 4}, keep_dims, ReductionType::Mean),
        ReductionParams(PartialShape{3, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6},
                        std::vector<int64_t>{1}, std::vector<T>{1.5, 3.5, 5.5}, keep_dims, ReductionType::Mean)};
    return params;
}

template <element::Type_t IN_ET,
          typename std::enable_if<std::is_integral<typename element_type_traits<IN_ET>::value_type>::value, bool>::type = true>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {
        ReductionParams(PartialShape{2, 2}, IN_ET, std::vector<T>{1, 2, 3, 4},
                        std::vector<int64_t>{0, 1}, std::vector<T>{2}, keep_dims, ReductionType::Mean),
        ReductionParams(PartialShape{3, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6},
                        std::vector<int64_t>{0}, std::vector<T>{3, 4}, keep_dims, ReductionType::Mean),
        ReductionParams(PartialShape{3, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6},
                        std::vector<int64_t>{1}, std::vector<T>{1, 3, 5}, keep_dims, ReductionType::Mean)};
    return params;
}

std::vector<ReductionParams> generateReductionCombinedParams() {
    const std::vector<std::vector<ReductionParams>> reductionTypeParams {
        generateReductionParams<element::Type_t::f32>(true),
        generateReductionParams<element::Type_t::f32>(false),
        generateReductionParams<element::Type_t::f16>(true),
        generateReductionParams<element::Type_t::f16>(false),
        generateReductionParams<element::Type_t::i32>(true),
        generateReductionParams<element::Type_t::i32>(false),
        generateReductionParams<element::Type_t::u32>(true),
        generateReductionParams<element::Type_t::u32>(false),
        generateReductionParams<element::Type_t::u64>(true),
        generateReductionParams<element::Type_t::u64>(false)
    };
    std::vector<ReductionParams> combinedParams;

    for (const auto& params : reductionTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

} // namespace
INSTANTIATE_TEST_SUITE_P(smoke_Reduction_With_Hardcoded_Refs, ReferenceReductionLayerTest, ::testing::ValuesIn(generateReductionCombinedParams()),
                         ReferenceReductionLayerTest::getTestCaseName);
} // namespace ReductionOpsRefTestDefinitions
} // namespace reference_tests

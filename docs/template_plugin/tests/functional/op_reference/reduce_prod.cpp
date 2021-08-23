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

template <element::Type_t IN_ET>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {
        ReductionParams(PartialShape{2, 2}, IN_ET, std::vector<T>{1, 2, 3, 4},
                        std::vector<int64_t>{0, 1}, std::vector<T>{24}, keep_dims, ReductionType::Prod),
        ReductionParams(PartialShape{3, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6},
                        std::vector<int64_t>{0}, std::vector<T>{15, 48}, keep_dims, ReductionType::Prod),
        ReductionParams(PartialShape{3, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6},
                        std::vector<int64_t>{1}, std::vector<T>{2, 12, 30}, keep_dims, ReductionType::Prod),
        ReductionParams(PartialShape{3, 3, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                        std::vector<int64_t>{0}, std::vector<T>{1 * 10 * 19,
                                                                2 * 11 * 20,
                                                                3 * 12 * 21,
                                                                4 * 13 * 22,
                                                                5 * 14 * 23,
                                                                6 * 15 * 24,
                                                                7 * 16 * 25,
                                                                8 * 17 * 26,
                                                                9 * 18 * 27},
                        keep_dims, ReductionType::Prod),
        ReductionParams(PartialShape{3, 3, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                        std::vector<int64_t>{2}, std::vector<T>{1 * 2 * 3,
                                                                4 * 5 * 6,
                                                                7 * 8 * 9,
                                                                10 * 11 * 12,
                                                                13 * 14 * 15,
                                                                16 * 17 * 18,
                                                                19 * 20 * 21,
                                                                22 * 23 * 24,
                                                                25 * 26 * 27},
                        keep_dims, ReductionType::Prod)
    };
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

    combinedParams.push_back(ReductionParams(PartialShape{3, 3, 3}, element::Type_t::f32, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                        std::vector<int64_t>{0, 1}, std::vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                                                                   2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                                                                   3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f},
                        true, ReductionType::Prod));
    combinedParams.push_back(ReductionParams(PartialShape{3, 3, 3}, element::Type_t::f32, std::vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                                      13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1},
                        std::vector<int64_t>{0, 1, 2}, std::vector<float>{1.0f * 10.0f * 9.0f * 4.0f * 13.0f * 6.0f * 7.0f *
                                                                      12.0f * 3.0f * 2.0f * 11.0f * 8.0f * 5.0f * 14.0f *
                                                                      5.0f * 8.0f * 11.0f * 2.0f * 3.0f * 12.0f * 7.0f *
                                                                      6.0f * 13.0f * 4.0f * 9.0f * 10.0f * 1.0f},
                        true, ReductionType::Prod));

    return combinedParams;
}
} // namespace
INSTANTIATE_TEST_SUITE_P(smoke_Reduction_With_Hardcoded_Refs, ReferenceReductionLayerTest, ::testing::ValuesIn(generateReductionCombinedParams()),
                         ReferenceReductionLayerTest::getTestCaseName);
} // namespace ReductionOpsRefTestDefinitions
} // namespace reference_tests

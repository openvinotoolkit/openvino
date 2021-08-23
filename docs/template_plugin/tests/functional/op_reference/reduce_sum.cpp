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

static std::mt19937_64 random_generator;

namespace reference_tests {
namespace ReductionOpsRefTestDefinitions {
namespace {

std::vector<float> generateRandomVector(const size_t count) {
    std::vector<float> result(count, 0);
    random_generator.seed(2);
    for (int i = 0; i < 1000000; i++) {
        result[i] = static_cast<float>(random_generator() % 255);
    }
    return result;
}

template <element::Type_t IN_ET>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {
        ReductionParams(PartialShape{2, 2}, IN_ET, std::vector<T>{1, 2, 3, 4},
                        std::vector<int64_t>{0, 1}, std::vector<T>{10}, keep_dims, ReductionType::Sum),
        ReductionParams(PartialShape{3, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6},
                        std::vector<int64_t>{0}, std::vector<T>{9, 12}, keep_dims, ReductionType::Sum),
        ReductionParams(PartialShape{3, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6},
                        std::vector<int64_t>{1}, std::vector<T>{3, 7, 11}, keep_dims, ReductionType::Sum),
        ReductionParams(PartialShape{3, 3, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                        std::vector<int64_t>{0}, std::vector<T>{1 + 10 + 19,
                                                                2 + 11 + 20,
                                                                3 + 12 + 21,
                                                                4 + 13 + 22,
                                                                5 + 14 + 23,
                                                                6 + 15 + 24,
                                                                7 + 16 + 25,
                                                                8 + 17 + 26,
                                                                9 + 18 + 27}, keep_dims, ReductionType::Sum),
        ReductionParams(PartialShape{3, 3, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                        std::vector<int64_t>{2}, std::vector<T>{1 + 2 + 3,
                                                                4 + 5 + 6,
                                                                7 + 8 + 9,
                                                                10 + 11 + 12,
                                                                13 + 14 + 15,
                                                                16 + 17 + 18,
                                                                19 + 20 + 21,
                                                                22 + 23 + 24,
                                                                25 + 26 + 27}, keep_dims, ReductionType::Sum),
        ReductionParams(PartialShape{3, 3, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                        std::vector<int64_t>{0, 1}, std::vector<T>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                                                                   2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                                                                   3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}, keep_dims, ReductionType::Sum),
        ReductionParams(PartialShape{3, 3, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                        std::vector<int64_t>{0, 1, 2}, std::vector<T>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 + 8 +
                                                                      17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}, keep_dims, ReductionType::Sum),
        ReductionParams(PartialShape{3, 3, 3, 3, 3}, IN_ET, std::vector<T>(std::pow(3, 5), 1),
                        std::vector<int64_t>{0, 1, 2, 3, 4}, std::vector<T>{243}, keep_dims, ReductionType::Sum)
    };
    return params;
}

std::vector<ReductionParams> generateReductionParamsFloat(const bool keep_dims) {
    std::vector<float> in = generateRandomVector(1000000);
    float res = std::accumulate(std::begin(in), std::end(in), 0);
    std::vector<ReductionParams> params = {
        ReductionParams(PartialShape{1000000}, element::f32, in, std::vector<int64_t>{0}, std::vector<float>{res}, keep_dims, ReductionType::Sum),
        ReductionParams(PartialShape{20}, element::f32, std::vector<float>{10000000.0f, 0.9f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                                                           0.8f,        0.1f, 0.9f, 0.5f, 0.2f, 0.3f, 0.4f,
                                                                           0.5f,        0.6f, 0.7f, 0.8f, 0.9f, 0.1f}, std::vector<int64_t>{0},
                        std::vector<float>{10000010.2f}, keep_dims, ReductionType::Sum)
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
        generateReductionParams<element::Type_t::u64>(false),
        generateReductionParamsFloat(true),
        generateReductionParamsFloat(false)
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

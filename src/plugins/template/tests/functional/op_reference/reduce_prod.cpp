// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_prod.hpp"

#include <gtest/gtest.h>

#include "reduction.hpp"

using namespace ov;

namespace reference_tests {
namespace ReductionOpsRefTestDefinitions {
namespace {
template <element::Type_t IN_ET>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {
        ReductionParams(ReductionType::Prod,
                        keep_dims,
                        std::vector<int64_t>{0, 1},
                        reference_tests::Tensor({2, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4}),
                        reference_tests::Tensor(reduce(Shape{2, 2}, AxisSet{0, 1}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{24})),
        ReductionParams(ReductionType::Prod,
                        keep_dims,
                        std::vector<int64_t>{0},
                        reference_tests::Tensor({3, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(reduce(Shape{3, 2}, AxisSet{0}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{15, 48})),
        ReductionParams(ReductionType::Prod,
                        keep_dims,
                        std::vector<int64_t>{1},
                        reference_tests::Tensor({3, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(reduce(Shape{3, 2}, AxisSet{1}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{2, 12, 30})),
        ReductionParams(ReductionType::Prod,
                        keep_dims,
                        std::vector<int64_t>{0},
                        reference_tests::Tensor({3, 3, 3},
                                                element::Type(IN_ET),
                                                std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}),
                        reference_tests::Tensor(reduce(Shape{3, 3, 3}, AxisSet{0}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1 * 10 * 19,
                                                               2 * 11 * 20,
                                                               3 * 12 * 21,
                                                               4 * 13 * 22,
                                                               5 * 14 * 23,
                                                               6 * 15 * 24,
                                                               7 * 16 * 25,
                                                               8 * 17 * 26,
                                                               9 * 18 * 27})),
        ReductionParams(ReductionType::Prod,
                        keep_dims,
                        std::vector<int64_t>{2},
                        reference_tests::Tensor({3, 3, 3},
                                                element::Type(IN_ET),
                                                std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}),
                        reference_tests::Tensor(reduce(Shape{3, 3, 3}, AxisSet{2}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1 * 2 * 3,
                                                               4 * 5 * 6,
                                                               7 * 8 * 9,
                                                               10 * 11 * 12,
                                                               13 * 14 * 15,
                                                               16 * 17 * 18,
                                                               19 * 20 * 21,
                                                               22 * 23 * 24,
                                                               25 * 26 * 27}))};
    auto out_shape_from_empty = Shape{2, 1, 1};
    if (keep_dims == false) {
        out_shape_from_empty = Shape{2};
    }
    const T default_val = T{1};
    params.push_back(ReductionParams(
        ReductionType::Prod,
        keep_dims,
        std::vector<int64_t>{1, 2},
        reference_tests::Tensor({2, 0, 4}, element::Type(IN_ET), std::vector<T>{}),
        reference_tests::Tensor(out_shape_from_empty, element::Type(IN_ET), std::vector<T>{default_val, default_val})));

    out_shape_from_empty = Shape{2, 0, 1};
    if (keep_dims == false) {
        out_shape_from_empty = Shape{2, 0};
    }
    params.push_back(
        ReductionParams(ReductionType::Prod,
                        keep_dims,
                        std::vector<int64_t>{2},
                        reference_tests::Tensor({2, 0, 4}, element::Type(IN_ET), std::vector<T>{}),
                        reference_tests::Tensor(out_shape_from_empty, element::Type(IN_ET), std::vector<T>{})));
    return params;
}

std::vector<ReductionParams> generateReductionCombinedParams() {
    const std::vector<std::vector<ReductionParams>> reductionTypeParams{
        generateReductionParams<element::Type_t::f32>(true),
        generateReductionParams<element::Type_t::f32>(false),
        generateReductionParams<element::Type_t::f16>(true),
        generateReductionParams<element::Type_t::f16>(false),
        generateReductionParams<element::Type_t::i32>(true),
        generateReductionParams<element::Type_t::i32>(false),
        generateReductionParams<element::Type_t::u32>(true),
        generateReductionParams<element::Type_t::u32>(false),
        generateReductionParams<element::Type_t::u64>(true),
        generateReductionParams<element::Type_t::u64>(false)};
    std::vector<ReductionParams> combinedParams;
    std::for_each(reductionTypeParams.begin(), reductionTypeParams.end(), [&](std::vector<ReductionParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });

    combinedParams.push_back(ReductionParams(
        ReductionType::Prod,
        true,
        std::vector<int64_t>{0, 1},
        reference_tests::Tensor({3, 3, 3},
                                element::Type_t::f32,
                                std::vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}),
        reference_tests::Tensor(
            {1, 1, 3},
            element::Type_t::f32,
            std::vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                               2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                               3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f})));
    combinedParams.push_back(ReductionParams(
        ReductionType::Prod,
        true,
        std::vector<int64_t>{0, 1, 2},
        reference_tests::Tensor({3, 3, 3}, element::Type_t::f32, std::vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,
                                                                                    10, 11, 12, 13, 14, 13, 12, 11, 10,
                                                                                    9,  8,  7,  6,  5,  4,  3,  2,  1}),
        reference_tests::Tensor({1, 1, 1},
                                element::Type_t::f32,
                                std::vector<float>{1.0f * 10.0f * 9.0f * 4.0f * 13.0f * 6.0f * 7.0f * 12.0f * 3.0f *
                                                   2.0f * 11.0f * 8.0f * 5.0f * 14.0f * 5.0f * 8.0f * 11.0f * 2.0f *
                                                   3.0f * 12.0f * 7.0f * 6.0f * 13.0f * 4.0f * 9.0f * 10.0f * 1.0f})));

    return combinedParams;
}
}  // namespace
INSTANTIATE_TEST_SUITE_P(smoke_Reduction_With_Hardcoded_Refs,
                         ReferenceReductionLayerTest,
                         ::testing::ValuesIn(generateReductionCombinedParams()),
                         ReferenceReductionLayerTest::getTestCaseName);
}  // namespace ReductionOpsRefTestDefinitions
}  // namespace reference_tests

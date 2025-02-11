// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_min.hpp"

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
        ReductionParams(ReductionType::Min,
                        keep_dims,
                        std::vector<int64_t>{0, 1},
                        reference_tests::Tensor({2, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4}),
                        reference_tests::Tensor(reduce(Shape{2, 2}, AxisSet{0, 1}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1})),
        ReductionParams(ReductionType::Min,
                        keep_dims,
                        std::vector<int64_t>{0},
                        reference_tests::Tensor({3, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(reduce(Shape{3, 2}, AxisSet{0}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1, 2})),
        ReductionParams(ReductionType::Min,
                        keep_dims,
                        std::vector<int64_t>{1},
                        reference_tests::Tensor({3, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(reduce(Shape{3, 2}, AxisSet{1}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1, 3, 5})),
        ReductionParams(ReductionType::Min,
                        keep_dims,
                        std::vector<int64_t>{0},
                        reference_tests::Tensor({3, 3, 3},
                                                element::Type(IN_ET),
                                                std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}),
                        reference_tests::Tensor(reduce(Shape{3, 3, 3}, AxisSet{0}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9})),
        ReductionParams(ReductionType::Min,
                        keep_dims,
                        std::vector<int64_t>{2},
                        reference_tests::Tensor({3, 3, 3},
                                                element::Type(IN_ET),
                                                std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}),
                        reference_tests::Tensor(reduce(Shape{3, 3, 3}, AxisSet{2}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1, 4, 7, 10, 13, 16, 19, 22, 25})),
        ReductionParams(ReductionType::Min,
                        keep_dims,
                        std::vector<int64_t>{0, 1},
                        reference_tests::Tensor({3, 3, 3},
                                                element::Type(IN_ET),
                                                std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}),
                        reference_tests::Tensor(reduce(Shape{3, 3, 3}, AxisSet{0, 1}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1, 2, 3})),
        ReductionParams(ReductionType::Min,
                        keep_dims,
                        std::vector<int64_t>{0, 1, 2},
                        reference_tests::Tensor({3, 3, 3},
                                                element::Type(IN_ET),
                                                std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}),
                        reference_tests::Tensor(reduce(Shape{3, 3, 3}, AxisSet{0, 1, 2}, keep_dims),
                                                element::Type(IN_ET),
                                                std::vector<T>{1}))};
    auto out_shape_from_empty = Shape{2, 1, 1};
    if (keep_dims == false) {
        out_shape_from_empty = Shape{2};
    }
    constexpr auto max_value =
        std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
    params.push_back(ReductionParams(
        ReductionType::Min,
        keep_dims,
        std::vector<int64_t>{1, 2},
        reference_tests::Tensor({2, 0, 4}, element::Type(IN_ET), std::vector<T>{}),
        reference_tests::Tensor(out_shape_from_empty, element::Type(IN_ET), std::vector<T>{max_value, max_value})));

    out_shape_from_empty = Shape{2, 0, 1};
    if (keep_dims == false) {
        out_shape_from_empty = Shape{2, 0};
    }
    params.push_back(
        ReductionParams(ReductionType::Min,
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
    return combinedParams;
}
}  // namespace
INSTANTIATE_TEST_SUITE_P(smoke_Reduction_With_Hardcoded_Refs,
                         ReferenceReductionLayerTest,
                         ::testing::ValuesIn(generateReductionCombinedParams()),
                         ReferenceReductionLayerTest::getTestCaseName);
}  // namespace ReductionOpsRefTestDefinitions
}  // namespace reference_tests

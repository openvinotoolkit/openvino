// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_l2.hpp"

#include <gtest/gtest.h>

#include "reduction.hpp"

using namespace ov;

namespace reference_tests {
namespace ReductionOpsRefTestDefinitions {
namespace {

template <element::Type_t IN_ET,
          typename std::enable_if<!std::is_integral<typename element_type_traits<IN_ET>::value_type>::value,
                                  bool>::type = true>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {ReductionParams(
        ReductionType::L2,
        keep_dims,
        std::vector<int64_t>{2},
        reference_tests::Tensor({3, 2, 2},
                                element::Type(IN_ET),
                                std::vector<T>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}),
        reference_tests::Tensor(reduce(Shape{3, 2, 2}, AxisSet{2}, keep_dims),
                                element::Type(IN_ET),
                                std::vector<T>{2.23606798, 5.0, 7.81024968, 10.63014581, 13.45362405, 16.2788206}))};
    auto out_shape_from_empty = Shape{2, 1, 1};
    if (keep_dims == false) {
        out_shape_from_empty = Shape{2};
    }
    params.push_back(
        ReductionParams(ReductionType::L2,
                        keep_dims,
                        std::vector<int64_t>{1, 2},
                        reference_tests::Tensor({2, 0, 4}, element::Type(IN_ET), std::vector<T>{}),
                        reference_tests::Tensor(out_shape_from_empty, element::Type(IN_ET), std::vector<T>{0, 0})));

    out_shape_from_empty = Shape{2, 0, 1};
    if (keep_dims == false) {
        out_shape_from_empty = Shape{2, 0};
    }
    params.push_back(
        ReductionParams(ReductionType::L2,
                        keep_dims,
                        std::vector<int64_t>{2},
                        reference_tests::Tensor({2, 0, 4}, element::Type(IN_ET), std::vector<T>{}),
                        reference_tests::Tensor(out_shape_from_empty, element::Type(IN_ET), std::vector<T>{})));
    return params;
}

template <element::Type_t IN_ET,
          typename std::enable_if<std::is_integral<typename element_type_traits<IN_ET>::value_type>::value,
                                  bool>::type = true>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {ReductionParams(
        ReductionType::L2,
        keep_dims,
        std::vector<int64_t>{2},
        reference_tests::Tensor({3, 2, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
        reference_tests::Tensor(reduce(Shape{3, 2, 2}, AxisSet{2}, keep_dims),
                                element::Type(IN_ET),
                                std::vector<T>{2, 5, 8, 11, 13, 16}))};

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

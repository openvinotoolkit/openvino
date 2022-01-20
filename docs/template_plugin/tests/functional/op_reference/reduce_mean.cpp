// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/reduce_mean.hpp"
#include "reduction.hpp"

using namespace ov;
using ReductionType = ngraph::helpers::ReductionType;

namespace reference_tests {
namespace ReductionOpsRefTestDefinitions {
namespace {

template <element::Type_t IN_ET,
          typename std::enable_if<!std::is_integral<typename element_type_traits<IN_ET>::value_type>::value, bool>::type = true>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {
        ReductionParams(ReductionType::Mean, keep_dims, std::vector<int64_t>{0, 1},
                        reference_tests::Tensor({2, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4}),
                        reference_tests::Tensor(ngraph::reduce(Shape{2, 2}, AxisSet{0, 1}, keep_dims), element::Type(IN_ET), std::vector<T>{2.5})),
        ReductionParams(ReductionType::Mean, keep_dims, std::vector<int64_t>{0},
                        reference_tests::Tensor({3, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(ngraph::reduce(Shape{3, 2}, AxisSet{0}, keep_dims), element::Type(IN_ET), std::vector<T>{3, 4})),
        ReductionParams(ReductionType::Mean, keep_dims, std::vector<int64_t>{1},
                        reference_tests::Tensor({3, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(ngraph::reduce(Shape{3, 2}, AxisSet{1}, keep_dims), element::Type(IN_ET), std::vector<T>{1.5, 3.5, 5.5}))
    };
    return params;
}

template <element::Type_t IN_ET,
          typename std::enable_if<std::is_integral<typename element_type_traits<IN_ET>::value_type>::value, bool>::type = true>
std::vector<ReductionParams> generateReductionParams(const bool keep_dims) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReductionParams> params = {
        ReductionParams(ReductionType::Mean, keep_dims, std::vector<int64_t>{0, 1},
                        reference_tests::Tensor({2, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4}),
                        reference_tests::Tensor(ngraph::reduce(Shape{2, 2}, AxisSet{0, 1}, keep_dims), element::Type(IN_ET), std::vector<T>{2})),
        ReductionParams(ReductionType::Mean, keep_dims, std::vector<int64_t>{0},
                        reference_tests::Tensor({3, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(ngraph::reduce(Shape{3, 2}, AxisSet{0}, keep_dims), element::Type(IN_ET), std::vector<T>{3, 4})),
        ReductionParams(ReductionType::Mean, keep_dims, std::vector<int64_t>{1},
                        reference_tests::Tensor({3, 2}, element::Type(IN_ET), std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(ngraph::reduce(Shape{3, 2}, AxisSet{1}, keep_dims), element::Type(IN_ET), std::vector<T>{1, 3, 5}))
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
    std::for_each(reductionTypeParams.begin(), reductionTypeParams.end(), [&](std::vector<ReductionParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}
} // namespace
INSTANTIATE_TEST_SUITE_P(smoke_Reduction_With_Hardcoded_Refs, ReferenceReductionLayerTest, ::testing::ValuesIn(generateReductionCombinedParams()),
                         ReferenceReductionLayerTest::getTestCaseName);
} // namespace ReductionOpsRefTestDefinitions
} // namespace reference_tests

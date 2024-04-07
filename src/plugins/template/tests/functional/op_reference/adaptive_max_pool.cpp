// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/adaptive_max_pool.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct AdaptiveMaxPoolParams {
    template <class IT>
    AdaptiveMaxPoolParams(const Shape& input_shape,
                          const Shape& output_shape,
                          const element::Type& input_type,
                          const element::Type& output_type,
                          const std::vector<IT>& input_values,
                          const std::vector<IT>& output_values,
                          const std::vector<int64_t>& output_indices,
                          const Shape& adaptive_shape,
                          const std::vector<int64_t>& adaptive_values)
        : m_input_shape(input_shape),
          m_output_shape(output_shape),
          m_input_type(input_type),
          m_output_type(output_type),
          m_input_data(CreateTensor(input_shape, input_type, input_values)),
          m_expected_data(CreateTensor(output_shape, output_type, output_values)),
          m_expected_indices(CreateTensor(output_shape, element::Type_t::i64, output_indices)),
          m_adaptive_shape(adaptive_shape),
          m_adaptive_values(adaptive_values) {}
    Shape m_input_shape;
    Shape m_output_shape;
    element::Type m_input_type;
    element::Type m_output_type;
    ov::Tensor m_input_data;
    ov::Tensor m_expected_data;
    ov::Tensor m_expected_indices;
    Shape m_adaptive_shape;
    std::vector<int64_t> m_adaptive_values;
};

class ReferenceAdaptiveMaxPoolLayerTest : public testing::TestWithParam<AdaptiveMaxPoolParams>,
                                          public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_input_shape,
                                  params.m_input_type,
                                  params.m_adaptive_shape,
                                  params.m_adaptive_values);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data, params.m_expected_indices};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AdaptiveMaxPoolParams>& obj) {
        auto params = obj.param;
        std::ostringstream result;
        result << "shape=" << params.m_input_shape << "_";
        result << "iType=" << params.m_input_type << "_";
        result << "shape=" << params.m_output_shape << "_";
        result << "oType=" << params.m_output_type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const Shape& adaptive_shape,
                                                 const std::vector<int64_t> adaptive_values) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto out = op::v0::Constant::create<int64_t>(element::Type_t::i64, adaptive_shape, adaptive_values);
        const auto adaptive_max_pool = std::make_shared<op::v8::AdaptiveMaxPool>(in, out);
        return std::make_shared<Model>(adaptive_max_pool->outputs(), ParameterVector{in});
    }
};

TEST_P(ReferenceAdaptiveMaxPoolLayerTest, AdaptiveMaxPoolWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<AdaptiveMaxPoolParams> generateParamsForAdaptiveMaxPoolWithExpectedResult() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AdaptiveMaxPoolParams> params{
        AdaptiveMaxPoolParams(
            ov::Shape{2, 3, 7},
            ov::Shape{2, 3, 3},
            IN_ET,
            IN_ET,
            std::vector<T>{0,  4,  1, 3, -2, -5, -2, -2, 1, -3, 1,  -3, -4, 0,  -2, 1, -1, -2, 3, -1, -3,
                           -1, -2, 3, 4, -3, -4, 1,  2,  0, -4, -5, -2, -2, -3, 2,  3, 1,  -5, 2, -4, -2},
            std::vector<T>{4,
                           3,
                           -2,
                           1,
                           1,
                           0,
                           1,
                           3,
                           3,

                           3,
                           4,
                           1,
                           2,
                           -2,
                           -2,
                           3,
                           2,
                           2},
            std::vector<int64_t>{1,
                                 3,
                                 4,
                                 1,
                                 3,
                                 6,
                                 1,
                                 4,
                                 4,

                                 2,
                                 3,
                                 6,
                                 0,
                                 4,
                                 4,
                                 1,
                                 4,
                                 4},
            ov::Shape{1},
            std::vector<int64_t>{3}),
        AdaptiveMaxPoolParams(
            ov::Shape{1, 3, 7, 10},
            ov::Shape{1, 3, 3, 3},
            IN_ET,
            IN_ET,
            std::vector<T>{
                0,  -2, -5, -5, 2,  3,  2,  -3, 1,  -2, -4, -1, -1, -1, 2,  -4, 3,  -5, -1, -1, 1,  2,  4,  -2,
                -3, -2, 0,  -5, 2,  -4, -1, -4, 4,  2,  1,  -2, 2,  -3, 0,  1,  -3, 3,  -1, 4,  0,  2,  0,  3,
                4,  -4, 1,  4,  -1, -5, -2, 4,  -3, 3,  2,  1,  0,  4,  2,  -5, 2,  -5, -2, -1, 4,  2,

                0,  4,  -2, 0,  -5, -3, 4,  -4, -2, -2, 2,  1,  4,  3,  2,  -5, -4, -4, 0,  1,  4,  -4, -3, 3,
                3,  4,  -2, -3, -4, -2, 0,  1,  -1, 3,  -2, 2,  0,  -3, -1, -1, 0,  0,  2,  2,  -2, 1,  -3, 1,
                2,  4,  3,  -5, -4, 1,  -4, 2,  0,  -2, -5, 2,  -3, -2, -3, -4, 2,  -2, -4, 2,  -4, -3,

                1,  -5, -1, -5, 2,  1,  3,  4,  3,  0,  -5, 4,  -3, -4, -1, 2,  -4, 2,  0,  -5, -3, 0,  2,  -3,
                -5, 3,  -2, -1, -5, -4, -5, 0,  -5, -1, -3, 3,  3,  -4, -3, -4, -5, 4,  -1, 1,  -1, -4, 1,  -3,
                -4, -1, -2, -3, -5, 2,  2,  -5, 1,  1,  -5, -4, 0,  2,  4,  2,  0,  2,  4,  0,  -5, 2},
            std::vector<T>{4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 4, 4, 3, 4, 4, 3, 3, 4, 4, 4},
            std::vector<int64_t>{22, 5,  16, 22, 43, 48, 43, 43, 48, 1,  6,  6,  20, 25,
                                 49, 50, 43, 49, 11, 6,  7,  41, 25, 36, 41, 66, 66},
            ov::Shape{2},
            std::vector<int64_t>{3, 3}),
        AdaptiveMaxPoolParams(ov::Shape{2, 2, 3, 3, 3},
                              ov::Shape{2, 2, 2, 2, 2},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{-5, 1,  -3, -4, 4,  -4, 3,  -3, -1, 0,  0,  -2, -4, 2,
                                             0,  -4, -5, -2, -4, -4, 0,  -2, 3,  -3, 4,  -1, -4,

                                             -1, -1, -5, 4,  -1, -2, -3, 0,  4,  -1, -5, -4, 1,  1,
                                             4,  -5, -5, -5, 4,  -3, -3, -3, 4,  0,  -3, -5, 1,

                                             4,  2,  1,  -5, -5, 1,  0,  -4, -1, 2,  -4, -2, 4,  3,
                                             1,  -3, -3, -2, -4, -3, -3, 3,  -1, 1,  2,  2,  -4,

                                             -5, -4, 1,  3,  -4, -1, 2,  4,  -5, 0,  1,  -2, 0,  0,
                                             -2, 3,  -2, -5, -3, -5, -2, -1, 3,  -2, 4,  3,  -3},
                              std::vector<T>{4, 4, 4, 4, 3, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4,
                                             4, 3, 4, 3, 4, 3, 4, 3, 3, 1, 4, 4, 3, 3, 4, 3},
                              std::vector<int64_t>{4, 4,  4,  4,  22, 22, 24, 22, 3, 14, 3, 8, 18, 14, 22, 14,
                                                   0, 13, 12, 13, 12, 13, 12, 13, 3, 2,  7, 7, 22, 22, 24, 22},
                              ov::Shape{3},
                              std::vector<int64_t>{2, 2, 2})};
    return params;
}

std::vector<AdaptiveMaxPoolParams> generateCombinedParamsForAdaptiveMaxPool() {
    const std::vector<std::vector<AdaptiveMaxPoolParams>> allTypeParams{
        generateParamsForAdaptiveMaxPoolWithExpectedResult<element::Type_t::f32>(),
        generateParamsForAdaptiveMaxPoolWithExpectedResult<element::Type_t::f16>(),
        generateParamsForAdaptiveMaxPoolWithExpectedResult<element::Type_t::bf16>(),
        generateParamsForAdaptiveMaxPoolWithExpectedResult<element::Type_t::i64>(),
        generateParamsForAdaptiveMaxPoolWithExpectedResult<element::Type_t::i32>(),
        generateParamsForAdaptiveMaxPoolWithExpectedResult<element::Type_t::i16>(),
        generateParamsForAdaptiveMaxPoolWithExpectedResult<element::Type_t::i8>(),
    };

    std::vector<AdaptiveMaxPoolParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_AdaptiveMaxPool_With_Hardcoded_Refs,
                         ReferenceAdaptiveMaxPoolLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForAdaptiveMaxPool()),
                         ReferenceAdaptiveMaxPoolLayerTest::getTestCaseName);

}  // namespace

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/adaptive_avg_pool.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct AdaptiveAvgPoolParams {
    template <class IT>
    AdaptiveAvgPoolParams(const Shape& input_shape,
                          const Shape& output_shape,
                          const element::Type& input_type,
                          const element::Type& ouput_type,
                          const std::vector<IT>& input_values,
                          const std::vector<IT>& output_values,
                          const Shape& adaptive_shape,
                          const std::vector<int64_t>& adaptive_values)
        : m_input_shape(input_shape),
          m_output_shape(output_shape),
          m_input_type(input_type),
          m_output_type(ouput_type),
          m_input_data(CreateTensor(m_input_shape, input_type, input_values)),
          m_expected_data(CreateTensor(m_output_shape, ouput_type, output_values)),
          m_adaptive_shape(adaptive_shape),
          m_adaptive_values(adaptive_values) {}
    Shape m_input_shape;
    Shape m_output_shape;
    element::Type m_input_type;
    element::Type m_output_type;
    ov::Tensor m_input_data;
    ov::Tensor m_expected_data;
    Shape m_adaptive_shape;
    std::vector<int64_t> m_adaptive_values;
};

class ReferenceAdaptiveAvgPoolLayerTest : public testing::TestWithParam<AdaptiveAvgPoolParams>,
                                          public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_input_shape,
                                  params.m_input_type,
                                  params.m_adaptive_shape,
                                  params.m_adaptive_values);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AdaptiveAvgPoolParams>& obj) {
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
        const auto adaptive_avg_pool = std::make_shared<op::v8::AdaptiveAvgPool>(in, out);
        return std::make_shared<Model>(NodeVector{adaptive_avg_pool}, ParameterVector{in});
    }
};

TEST_P(ReferenceAdaptiveAvgPoolLayerTest, AdaptiveAvgPoolWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<AdaptiveAvgPoolParams> generateParamsForAdaptiveAvgPool() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AdaptiveAvgPoolParams> params{
        AdaptiveAvgPoolParams(
            ov::Shape{2, 3, 7},
            ov::Shape{2, 3, 3},
            IN_ET,
            IN_ET,
            std::vector<T>{0,  4,  1, 3, -2, -5, -2, -2, 1, -3, 1,  -3, -4, 0,  -2, 1, -1, -2, 3, -1, -3,
                           -1, -2, 3, 4, -3, -4, 1,  2,  0, -4, -5, -2, -2, -3, 2,  3, 1,  -5, 2, -4, -2},
            std::vector<T>{1.66666663,
                           0.66666669,
                           -3.,
                           -1.33333337,
                           -1.66666663,
                           -2.33333325,
                           -0.66666669,
                           0.,
                           -0.33333334,

                           0.,
                           1.33333337,
                           -2.,
                           -0.66666669,
                           -3.66666675,
                           -2.33333325,
                           2.,
                           -0.66666669,
                           -1.33333337},
            ov::Shape{1},
            {3}),
        AdaptiveAvgPoolParams(
            ov::Shape{1, 3, 7, 10},
            ov::Shape{1, 3, 3, 3},
            IN_ET,
            IN_ET,
            std::vector<T>{
                -2, -3, -4, 3,  -5, 4,  0,  -4, -2, -4, -5, 0,  -3, 0,  -2, 0,  0,  -5, -4, -1, 3,  -1, 0,  -1,
                0,  -2, 0,  4,  1,  4,  0,  -1, -4, 2,  -2, -5, -1, -1, -2, 1,  2,  -2, -1, 2,  0,  -1, 0,  -5,
                4,  4,  3,  0,  -4, -4, -4, -2, 0,  1,  -2, -1, 4,  -2, -4, 1,  -1, -3, -4, -1, 1,  -4,

                -2, -4, -5, 0,  -4, 3,  4,  -5, -4, -2, 0,  2,  -4, -3, 3,  -1, 1,  -4, -5, 4,  2,  -5, 2,  -3,
                0,  4,  3,  3,  1,  2,  -1, -4, 1,  -3, -3, -2, 3,  4,  -2, -5, 1,  4,  4,  -2, 2,  1,  -5, -2,
                -5, 1,  1,  -2, -3, -3, -1, -5, 1,  -3, -5, -3, -4, -1, 4,  -3, 4,  -1, 4,  3,  1,  4,

                -2, -4, -4, 4,  -3, 4,  2,  -3, -2, 4,  -3, 0,  1,  -4, 4,  4,  0,  3,  -1, 3,  3,  -5, 0,  3,
                -3, 1,  -2, 4,  -5, -5, 1,  0,  -1, 0,  -3, -2, 0,  -3, 3,  -2, -2, 0,  -3, 4,  -1, 2,  -2, 2,
                -3, -1, -4, -2, 0,  2,  0,  2,  0,  -3, 4,  3,  -5, -3, -5, 1,  -5, -3, -5, 4,  -3, 3},
            std::vector<T>{-1.08333337, -0.25000000, -0.91666669, -0.08333334, -0.66666669,
                           0.75000000,  -0.41666666, -1.33333337, -0.58333331,

                           -1.66666663, 0.58333331,  -0.16666667, -0.33333334, -0.41666666,
                           -0.16666667, -0.33333334, -0.66666669, -0.75000000,

                           -0.91666669, 0.83333331,  -0.16666667, 0.,          -0.25000000,
                           -1.16666663, -1.41666663, -0.41666666, -0.08333334},
            ov::Shape{2},
            {3, 3}),
        AdaptiveAvgPoolParams(
            ov::Shape{2, 2, 3, 3, 3},
            ov::Shape{2, 2, 2, 2, 2},
            IN_ET,
            IN_ET,
            std::vector<T>{-5, 1,  -3, -4, 4,  -4, 3,  -3, -1, 0,  0,  -2, -4, 2,  0,  -4, -5, -2, -4, -4, 0,  -2,
                           3,  -3, 4,  -1, -4, -1, -1, -5, 4,  -1, -2, -3, 0,  4,  -1, -5, -4, 1,  1,  4,  -5, -5,
                           -5, 4,  -3, -3, -3, 4,  0,  -3, -5, 1,  4,  2,  1,  -5, -5, 1,  0,  -4, -1, 2,  -4, -2,
                           4,  3,  1,  -3, -3, -2, -4, -3, -3, 3,  -1, 1,  2,  2,  -4, -5, -4, 1,  3,  -4, -1, 2,
                           4,  -5, 0,  1,  -2, 0,  0,  -2, 3,  -2, -5, -3, -5, -2, -1, 3,  -2, 4,  3,  -3},
            std::vector<T>{-0.750, -0.250, -1.375, -1.125, -1.125, -0.500, -0.875, -1.250, -0.375, -1.625, -1.,
                           -0.500, -0.250, -0.750, -1.875, -0.625, 0.125,  -0.375, -1.625, -1.250, 0.,     -1.,
                           0.875,  -0.375, -1.125, -1.375, 0.750,  -1.875, -0.625, -1.125, 1.250,  -1.},
            ov::Shape{3},
            {2, 2, 2}),
    };
    return params;
}

std::vector<AdaptiveAvgPoolParams> generateCombinedParamsForAdaptiveAvgPool() {
    const std::vector<std::vector<AdaptiveAvgPoolParams>> allTypeParams{
        generateParamsForAdaptiveAvgPool<element::Type_t::f32>(),
        generateParamsForAdaptiveAvgPool<element::Type_t::f16>(),
        generateParamsForAdaptiveAvgPool<element::Type_t::bf16>()};

    std::vector<AdaptiveAvgPoolParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_AdaptiveAvgPool_With_Hardcoded_Refs,
                         ReferenceAdaptiveAvgPoolLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForAdaptiveAvgPool()),
                         ReferenceAdaptiveAvgPoolLayerTest::getTestCaseName);

}  // namespace

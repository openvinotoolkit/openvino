// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rms_norm.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "common_test_utils/common_utils.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace reference_tests;

struct RMSNormParams {
    RMSNormParams(const reference_tests::Tensor& paramInput,
                  const reference_tests::Tensor& paramReductionAxes,
                  const double eps,
                  const reference_tests::Tensor& paramExpected,
                  const reference_tests::Tensor& paramScale = {})
        : input(paramInput),
          reductionAxes(paramReductionAxes),
          eps(eps),
          expected(paramExpected) {
        if (paramScale.data) {
            scale = paramScale;
        }
    }
    reference_tests::Tensor input;
    reference_tests::Tensor reductionAxes;
    reference_tests::Tensor scale;
    double eps;
    reference_tests::Tensor expected;
};

class ReferenceRMSNormLayerTest : public testing::TestWithParam<RMSNormParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input, params.reductionAxes, params.eps, params.scale);
        if (!params.scale.data) {
            inputData = {params.input.data};
        } else {
            inputData = {params.input.data, params.scale.data};
        }
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RMSNormParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape;
        result << "_iType=" << param.input.type;
        result << "_reductionAxes="
               << ov::test::utils::vec2str(std::vector<int64_t>(
                      obj.param.reductionAxes.data.data<int64_t>(),
                      obj.param.reductionAxes.data.data<int64_t>() + shape_size(obj.param.reductionAxes.shape)));
        if (param.scale.data) {
            result << "_Scaled="
                   << "True";
        }
        result << "_eps=" << param.eps;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& input,
                                                 const reference_tests::Tensor& reductionAxes,
                                                 const double eps,
                                                 const reference_tests::Tensor& scale = {}) {
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        const auto axes = std::make_shared<op::v0::Constant>(reductionAxes.type,
                                                             reductionAxes.shape,
                                                             reductionAxes.data.data<int64_t>());

        if (!scale.data) {
            const auto rms_norm = std::make_shared<op::v14::RMSNorm>(in, axes, eps);
            return std::make_shared<ov::Model>(NodeVector{rms_norm}, ParameterVector{in});
        }
        const auto scale_param = std::make_shared<op::v0::Parameter>(scale.type, scale.shape);
        const auto rms_norm = std::make_shared<op::v14::RMSNorm>(in, axes, scale_param, eps);
        return std::make_shared<ov::Model>(NodeVector{rms_norm}, ParameterVector{in, scale_param});
    }
};

TEST_P(ReferenceRMSNormLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_RMSNorm_With_Hardcoded_Refs,
    ReferenceRMSNormLayerTest,
    ::testing::Values(
        RMSNormParams(reference_tests::Tensor{Shape{8},
                                              ov::element::f32,
                                              std::vector<float>({-6.44250308,
                                                                  -59.65135475,
                                                                  28.08134504,
                                                                  -3.38603289,
                                                                  1.047344,
                                                                  -22.62146978,
                                                                  58.72749089,
                                                                  16.00083578})},
                      reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
                      1e-5,
                      reference_tests::Tensor{Shape{8},
                                              ov::element::f32,
                                              std::vector<float>{-0.19629386,
                                                                 -1.81749151,
                                                                 0.85559844,
                                                                 -0.10316758,
                                                                 0.03191107,
                                                                 -0.68924385,
                                                                 1.7893427,
                                                                 0.48752259}}),
        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 3},
                ov::element::f32,
                std::vector<float>({-6.44250308, -59.65135475, 28.08134504, -3.38603289, 1.047344, -22.62146978})},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
            1e-5,
            reference_tests::Tensor{
                Shape{2, 3},
                ov::element::f32,
                std::vector<float>{-0.16844749, -1.559661, 0.7342227, -0.25613253, 0.07922512, -1.71117484}}),

        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 3, 1},
                ov::element::f32,
                std::vector<float>({-0.64425033, -5.9651356, 2.8081346, -0.3386033, 0.1047344, -2.262147})},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
            1e-5,
            reference_tests::Tensor{
                Shape{2, 3, 1},
                ov::element::f32,
                std::vector<float>{-0.99998795, -0.99999986, 0.99999937, -0.99995639, 0.99954449, -0.99999902}}),

        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 3, 1},
                ov::element::f32,
                std::vector<float>({-0.64425033, -5.9651356, 2.8081346, -0.3386033, 0.1047344, -2.262147})},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({1})},
            1e-5,
            reference_tests::Tensor{
                Shape{2, 3, 1},
                ov::element::f32,
                std::vector<float>{-0.16844743, -1.5596604, 0.7342224, -0.2561318, 0.0792249, -1.71117}}),
        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 3, 1},
                ov::element::f32,
                std::vector<float>({-0.64425033, -5.9651356, 2.8081346, -0.3386033, 0.1047344, -2.262147})},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({0})},
            1e-5,
            reference_tests::Tensor{Shape{2, 3, 1},
                                    ov::element::f32,
                                    std::vector<float>{-1.2518, -1.4140, 1.1013, -0.6579, 0.0248, -0.8872}}),

        RMSNormParams(reference_tests::Tensor{Shape{2, 2, 2, 3},
                                              ov::element::f32,
                                              std::vector<float>({-0.64425033, -5.9651356, 2.8081346,  -0.3386033,
                                                                  0.1047344,   -2.262147,  5.872749,   1.6000836,
                                                                  -6.754028,   4.015047,   9.291021,   0.00016722,
                                                                  7.7904015,   -3.167727,  1.3428825,  -1.4490807,
                                                                  -1.2650547,  5.5311837,  0.71208346, 9.074844,
                                                                  0.8841632,   -8.358102,  -2.673152,  7.01701})},
                      reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
                      1e-5,
                      reference_tests::Tensor{
                          Shape{2, 2, 2, 3},
                          ov::element::f32,
                          std::vector<float>{-0.16844743, -1.5596604, 0.7342224,  -0.2561318, 0.0792249,   -1.71117,
                                             1.1187618,   0.30481678, -1.2866459, 0.687082,   1.5899425,   0.00002862,
                                             1.5844078,   -0.6442507, 0.27311474, -0.4285907, -0.3741618,  1.6359433,
                                             0.1348591,   1.7186543,  0.1674487,  -1.288446,  -0.41208065, 1.0817096}}),

        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 2, 2, 3},
                ov::element::f32,
                std::vector<float>({-0.64425033, -5.9651356, 2.8081346, -0.3386033, 0.1047344,  -2.262147,
                                    5.872749,    1.6000836,  -6.754028, 4.015047,   9.291021,   0.00016722,
                                    7.7904015,   -3.167727,  1.3428825, -1.4490807, -1.2650547, 5.5311837,
                                    0.71208346,  9.074844,   0.8841632, -8.358102,  -2.673152,  7.01701})},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
            1e-5,
            reference_tests::Tensor{
                Shape{2, 2, 2, 3},
                ov::element::f32,
                std::vector<float>{-0.08422372, -0.77983022, 0.36711121,  -0.1280659,  0.03961245,  -0.85558498,
                                   0.55938089,  0.15240839,  -0.64332294, 0.343541,    0.79497123,  0.00001431,
                                   0.7922039,   -0.32212535, 0.13655737,  -0.21429534, -0.1870809,  0.81797165,
                                   0.06742955,  0.85932714,  0.08372435,  -0.64422297, -0.20604032, 0.54085481}},
            reference_tests::Tensor{Shape{1}, ov::element::f32, std::vector<float>{0.5}}),

        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 2, 2, 3},
                ov::element::f32,
                std::vector<float>({-0.64425033, -5.9651356, 2.8081346, -0.3386033, 0.1047344,  -2.262147,
                                    5.872749,    1.6000836,  -6.754028, 4.015047,   9.291021,   0.00016722,
                                    7.7904015,   -3.167727,  1.3428825, -1.4490807, -1.2650547, 5.5311837,
                                    0.71208346,  9.074844,   0.8841632, -8.358102,  -2.673152,  7.01701})},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({3})},
            1e-5,
            reference_tests::Tensor{
                Shape{2, 2, 2, 3},
                ov::element::f32,
                std::vector<float>{-0.08422372, -2.33949065, 0.1835556,   -0.1280659,  0.11883735,  -0.42779249,
                                   0.55938089,  0.45722517,  -0.32166147, 0.343541,    2.38491368,  0.00000715,
                                   0.7922039,   -0.96637604, 0.06827869,  -0.21429534, -0.56124271, 0.40898582,
                                   0.06742955,  2.57798141,  0.04186217,  -0.64422297, -0.61812097, 0.27042741}},
            reference_tests::Tensor{Shape{1, 3}, ov::element::f32, std::vector<float>{0.5, 1.5, 0.25}}),

        RMSNormParams(reference_tests::Tensor{Shape{1, 3, 3, 3},
                                              ov::element::f32,
                                              std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5,
                                                                  6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
                      reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({3})},
                      1e-5,
                      reference_tests::Tensor{
                          Shape{1, 3, 3, 3},
                          ov::element::f32,
                          std::vector<float>{0.46290955, 0.92581911, 1.38872866, 0.78954188, 0.98692735, 1.18431282,
                                             0.87047794, 0.99483193, 1.11918592, 0.46290955, 0.92581911, 1.38872866,
                                             0.78954188, 0.98692735, 1.18431282, 0.87047794, 0.99483193, 1.11918592,
                                             0.46290955, 0.92581911, 1.38872866, 0.78954188, 0.98692735, 1.18431282,
                                             0.87047794, 0.99483193, 1.11918592}})),
    ReferenceRMSNormLayerTest::getTestCaseName);

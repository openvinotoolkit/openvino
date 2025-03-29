// Copyright (C) 2018-2025 Intel Corporation
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
            inputData = {params.input.data, params.reductionAxes.data};
        } else {
            inputData = {params.input.data, params.reductionAxes.data, params.scale.data};
        }
        refOutData = {params.expected.data};
        if (params.input.type == ov::element::f32) {
            threshold = 1e-5f;  // Set more precise threshold to detect eps changes
        }
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RMSNormParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape;
        result << "_iType=" << param.input.type;
        result << "_axesType=" << param.reductionAxes.type;
        result << "_reductionAxes="
               << ov::test::utils::vec2str(op::v0::Constant(param.reductionAxes.data).cast_vector<int64_t>());
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
        const auto axes = std::make_shared<op::v0::Parameter>(reductionAxes.type, reductionAxes.shape);

        if (!scale.data) {
            const auto rms_norm = std::make_shared<op::internal::RMSNorm>(in, axes, eps);
            return std::make_shared<ov::Model>(NodeVector{rms_norm}, ParameterVector{in, axes});
        }
        const auto scale_param = std::make_shared<op::v0::Parameter>(scale.type, scale.shape);
        const auto rms_norm = std::make_shared<op::internal::RMSNorm>(in, axes, scale_param, eps);
        return std::make_shared<ov::Model>(NodeVector{rms_norm}, ParameterVector{in, axes, scale_param});
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
                      reference_tests::Tensor{Shape{1}, ov::element::i32, std::vector<int32_t>({-1})},
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
                      reference_tests::Tensor{Shape{1}, ov::element::i32, std::vector<int32_t>({-1})},
                      1e-2,
                      reference_tests::Tensor{Shape{8},
                                              ov::element::f32,
                                              std::vector<float>{-0.19629295,
                                                                 -1.81748319,
                                                                 0.85559446,
                                                                 -0.10316710,
                                                                 0.03191093,
                                                                 -0.68924063,
                                                                 1.78933442,
                                                                 0.48752034}}),
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
                      reference_tests::Tensor{Shape{1}, ov::element::i32, std::vector<int32_t>({-1})},
                      5.55,
                      reference_tests::Tensor{Shape{8},
                                              ov::element::f32,
                                              std::vector<float>{-0.19579013,
                                                                 -1.81282747,
                                                                 0.85340279,
                                                                 -0.10290283,
                                                                 0.03182918,
                                                                 -0.68747509,
                                                                 1.78475082,
                                                                 0.48627150}}),
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
            reference_tests::Tensor{
                Shape{2, 3, 1},
                ov::element::f32,
                std::vector<float>{-1.25182056, -1.41399515, 1.10131621, -0.65792835, 0.02482658, -0.88718647}}),
        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 3, 1},
                ov::element::f32,
                std::vector<float>({-0.64425033, -5.9651356, 2.8081346, -0.3386033, 0.1047344, -2.262147})},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({0})},
            1e-1,
            reference_tests::Tensor{
                Shape{2, 3, 1},
                ov::element::f32,
                std::vector<float>{-1.06658208, -1.41003966, 1.09294367, -0.56057125, 0.02475713, -0.88044184}}),

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
        RMSNormParams(reference_tests::Tensor{Shape{2, 2, 2, 3},
                                              ov::element::bf16,
                                              std::vector<ov::bfloat16>{
                                                  -0.6445, -5.9688, 2.8125, -0.3379, 0.1045, -2.2656, 5.8750,  1.6016,
                                                  -6.7500, 4.0000,  9.3125, 0.0002,  7.7812, -3.1719, 1.3438,  -1.4453,
                                                  -1.2656, 5.5312,  0.7109, 9.0625,  0.8828, -8.3750, -2.6719, 7.0312}},
                      reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
                      1e-5,
                      reference_tests::Tensor{
                          Shape{2, 2, 2, 3},
                          ov::element::bf16,
                          std::vector<ov::bfloat16>{-0.1680, -1.5625, 0.7344,  -0.2559, 0.0791,  -1.7188,
                                                    1.1172,  0.3047,  -1.2891, 0.6836,  1.5938,  0.0000,
                                                    1.5859,  -0.6484, 0.2734,  -0.4277, -0.3750, 1.6406,
                                                    0.1348,  1.7188,  0.1670,  -1.2891, -0.4102, 1.0781}}),
        RMSNormParams(reference_tests::Tensor{Shape{2, 2, 2, 3},
                                              ov::element::bf16,
                                              std::vector<ov::bfloat16>{
                                                  -0.6445, -5.9688, 2.8125, -0.3379, 0.1045, -2.2656, 5.8750,  1.6016,
                                                  -6.7500, 4.0000,  9.3125, 0.0002,  7.7812, -3.1719, 1.3438,  -1.4453,
                                                  -1.2656, 5.5312,  0.7109, 9.0625,  0.8828, -8.3750, -2.6719, 7.0312}},
                      reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
                      1e-5,
                      reference_tests::Tensor{Shape{2, 2, 2, 3},
                                              ov::element::bf16,
                                              std::vector<ov::bfloat16>{
                                                  -0.0840, -0.7812, 0.3672, -0.1279, 0.0396, -0.8594, 0.5586,  0.1523,
                                                  -0.6445, 0.3418,  0.7969, 0.0000,  0.7930, -0.3242, 0.1367,  -0.2139,
                                                  -0.1875, 0.8203,  0.0674, 0.8594,  0.0835, -0.6445, -0.2051, 0.5391}},
                      reference_tests::Tensor{Shape{1}, ov::element::bf16, std::vector<ov::bfloat16>{0.5}}),
        RMSNormParams(reference_tests::Tensor{Shape{2, 2, 2, 3},
                                              ov::element::f16,
                                              std::vector<ov::float16>{
                                                  -0.644, -5.965, 2.809, -0.3386,   0.10474, -2.262, 5.87,   1.6,
                                                  -6.754, 4.016,  9.29,  0.0001673, 7.79,    -3.168, 1.343,  -1.449,
                                                  -1.265, 5.53,   0.712, 9.08,      0.8843,  -8.36,  -2.674, 7.016}},
                      reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
                      1e-5,
                      reference_tests::Tensor{
                          Shape{2, 2, 2, 3},
                          ov::element::f16,
                          std::vector<ov::float16>{-0.1683, -1.559, 0.734,  -0.256,    0.0792, -1.711, 1.118,  0.3047,
                                                   -1.286,  0.687,  1.59,   0.0000286, 1.584,  -0.644, 0.273,  -0.4287,
                                                   -0.374,  1.636,  0.1348, 1.719,     0.1675, -1.288, -0.412, 1.081}}),
        RMSNormParams(reference_tests::Tensor{Shape{2, 2, 2, 3},
                                              ov::element::f16,
                                              std::vector<ov::float16>{
                                                  -0.644, -5.965, 2.809, -0.3386,   0.10474, -2.262, 5.87,   1.6,
                                                  -6.754, 4.016,  9.29,  0.0001673, 7.79,    -3.168, 1.343,  -1.449,
                                                  -1.265, 5.53,   0.712, 9.08,      0.8843,  -8.36,  -2.674, 7.016}},
                      reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
                      1e-5,
                      reference_tests::Tensor{
                          Shape{2, 2, 2, 3},
                          ov::element::f16,
                          std::vector<ov::float16>{

                              -0.08417, -0.7793, 0.367,  -0.128,    0.0396,  -0.8555, 0.559,  0.1523,
                              -0.643,   0.3435,  0.795,  0.0000143, 0.792,   -0.322,  0.1365, -0.2144,
                              -0.187,   0.818,   0.0674, 0.8594,    0.08374, -0.644,  -0.206, 0.5405}},
                      reference_tests::Tensor{Shape{1}, ov::element::f16, std::vector<ov::float16>{0.5}}),
        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 2, 2, 3},
                ov::element::f64,
                std::vector<double>{-0.64425031, -5.96513547, 2.8081345,   -0.33860329, 0.1047344,   -2.26214698,
                                    5.87274909,  1.60008358,  -6.75402803, 4.01504693,  9.2910216,   0.00016722,
                                    7.79040128,  -3.16772695, 1.34288255,  -1.44908073, -1.26505474, 5.5311837,
                                    0.71208347,  9.07484454,  0.8841632,   -8.35810155, -2.67315197, 7.01701008}},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
            1e-5,
            reference_tests::Tensor{
                Shape{2, 2, 2, 3},
                ov::element::f64,
                std::vector<double>{-0.16844743, -1.55966048, 0.73422245, -0.2561318,  0.0792249,   -1.71116999,
                                    1.1187618,   0.30481677,  -1.2866459, 0.68708204,  1.58994258,  0.00002862,
                                    1.58440782,  -0.64425068, 0.27311477, -0.42859069, -0.37416182, 1.63594325,
                                    0.1348591,   1.71865438,  0.1674487,  -1.28844602, -0.41208066, 1.0817096}}),
        RMSNormParams(
            reference_tests::Tensor{
                Shape{2, 2, 2, 3},
                ov::element::f64,
                std::vector<double>{-0.64425031, -5.96513547, 2.8081345,   -0.33860329, 0.1047344,   -2.26214698,
                                    5.87274909,  1.60008358,  -6.75402803, 4.01504693,  9.2910216,   0.00016722,
                                    7.79040128,  -3.16772695, 1.34288255,  -1.44908073, -1.26505474, 5.5311837,
                                    0.71208347,  9.07484454,  0.8841632,   -8.35810155, -2.67315197, 7.01701008}},
            reference_tests::Tensor{Shape{1}, ov::element::i64, std::vector<int64_t>({-1})},
            1e-5,
            reference_tests::Tensor{
                Shape{2, 2, 2, 3},
                ov::element::f64,
                std::vector<double>{-0.08422372, -0.77983024, 0.36711123,  -0.1280659,  0.03961245,  -0.855585,
                                    0.5593809,   0.15240838,  -0.64332295, 0.34354102,  0.79497129,  0.00001431,
                                    0.79220391,  -0.32212534, 0.13655738,  -0.21429535, -0.18708091, 0.81797163,
                                    0.06742955,  0.85932719,  0.08372435,  -0.64422301, -0.20604033, 0.5408548}},
            reference_tests::Tensor{Shape{1}, ov::element::f64, std::vector<double>{0.5}}),
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

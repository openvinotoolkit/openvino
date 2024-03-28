// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct BatchNormParams {
    template <class T>
    BatchNormParams(const Shape& input_shape,
                    const Shape& expected_shape,
                    const element::Type& input_type,
                    const element::Type& expected_type,
                    const std::vector<T>& input_value,
                    const std::vector<T>& expected_value,
                    const std::vector<T>& gamma,
                    const std::vector<T>& beta,
                    const std::vector<T>& mean,
                    const std::vector<T>& variance,
                    const float epsilon)
        : m_input_shape(input_shape),
          m_expected_shape(expected_shape),
          m_input_type(input_type),
          m_expected_type(expected_type),
          m_input_value(CreateTensor(input_shape, input_type, input_value)),
          m_expected_value(CreateTensor(expected_shape, expected_type, expected_value)),
          m_gamma(CreateTensor(Shape{input_shape.at(1)}, input_type, gamma)),
          m_beta(CreateTensor(Shape{input_shape.at(1)}, input_type, beta)),
          m_mean(CreateTensor(Shape{input_shape.at(1)}, input_type, mean)),
          m_variance(CreateTensor(Shape{input_shape.at(1)}, input_type, variance)),
          m_epsilon(epsilon) {}

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
    ov::Tensor m_gamma;
    ov::Tensor m_beta;
    ov::Tensor m_mean;
    ov::Tensor m_variance;
    float m_epsilon;
};

class ReferenceBatchNormV0LayerTest : public testing::TestWithParam<BatchNormParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type, params.m_epsilon);
        inputData = {params.m_input_value, params.m_gamma, params.m_beta, params.m_mean, params.m_variance};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BatchNormParams>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "output_type=" << param.m_expected_type << "; ";
        result << "gamma=" << param.m_gamma.data() << "; ";
        result << "beta=" << param.m_beta.data() << "; ";
        result << "mean=" << param.m_mean.data() << "; ";
        result << "variance=" << param.m_variance.data();

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type_t& input_type,
                                                 const float epsilon) {
        Shape channel_shape{input_shape.at(1)};
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        auto gamma = std::make_shared<op::v0::Parameter>(input_type, channel_shape);
        auto beta = std::make_shared<op::v0::Parameter>(input_type, channel_shape);
        auto mean = std::make_shared<op::v0::Parameter>(input_type, channel_shape);
        auto variance = std::make_shared<op::v0::Parameter>(input_type, channel_shape);
        auto batch_norm = std::make_shared<op::v0::BatchNormInference>(in, gamma, beta, mean, variance, epsilon);

        return std::make_shared<ov::Model>(batch_norm, ParameterVector{in, gamma, beta, mean, variance});
    }
};

class ReferenceBatchNormV5LayerTest : public ReferenceBatchNormV0LayerTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type, params.m_epsilon);
        inputData = {params.m_input_value, params.m_gamma, params.m_beta, params.m_mean, params.m_variance};
        refOutData = {params.m_expected_value};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type_t& input_type,
                                                 const float epsilon) {
        Shape channel_shape{input_shape.at(1)};
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        auto gamma = std::make_shared<op::v0::Parameter>(input_type, channel_shape);
        auto beta = std::make_shared<op::v0::Parameter>(input_type, channel_shape);
        auto mean = std::make_shared<op::v0::Parameter>(input_type, channel_shape);
        auto variance = std::make_shared<op::v0::Parameter>(input_type, channel_shape);
        auto batch_norm = std::make_shared<op::v5::BatchNormInference>(in, gamma, beta, mean, variance, epsilon);

        return std::make_shared<ov::Model>(batch_norm, ParameterVector{in, gamma, beta, mean, variance});
    }
};

TEST_P(ReferenceBatchNormV0LayerTest, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceBatchNormV5LayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<BatchNormParams> generateParamsForBatchNorm() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<BatchNormParams> params{
        /*------------- 2d --------------*/
        BatchNormParams(Shape{2, 3},
                        Shape{2, 3},
                        ET,
                        ET,
                        std::vector<T>{1, 2, 3, -1, -2, -3},
                        std::vector<T>{2, 6, 12, -2, -6, -12},
                        std::vector<T>{2.0, 3.0, 4.0},
                        std::vector<T>{0.0, 0.0, 0.0},
                        std::vector<T>{0.0, 0.0, 0.0},
                        std::vector<T>{0.75, 0.75, 0.75},
                        0.25),
        BatchNormParams(Shape{2, 3},
                        Shape{2, 3},
                        ET,
                        ET,
                        std::vector<T>{1, 2, 3, -1, -2, -3},
                        std::vector<T>{3, 0, 6, 1, -4, 0},
                        std::vector<T>{1.0, 1.0, 1.0},
                        std::vector<T>{2.0, -2.0, 3.0},
                        std::vector<T>{0.0, 0.0, 0.0},
                        std::vector<T>{0.75, 0.75, 0.75},
                        0.25),
        BatchNormParams(Shape{2, 3},
                        Shape{2, 3},
                        ET,
                        ET,
                        std::vector<T>{1, 2, 3, -1, -2, -3},
                        std::vector<T>{3, 0, 6, 1, -4, 0},
                        std::vector<T>{1.0, 1.0, 1.0},
                        std::vector<T>{0.0, 0.0, 0.0},
                        std::vector<T>{-2.0, 2.0, -3.0},
                        std::vector<T>{0.75, 0.75, 0.75},
                        0.25),
        BatchNormParams(Shape{2, 3},
                        Shape{2, 3},
                        ET,
                        ET,
                        std::vector<T>{3, 5, 1, -3, -5, -1},
                        std::vector<T>{2, 2, 2, -2, -2, -2},
                        std::vector<T>{1.0, 1.0, 1.0},
                        std::vector<T>{0.0, 0.0, 0.0},
                        std::vector<T>{0.0, 0.0, 0.0},
                        std::vector<T>{2.0, 6.0, 0.0},
                        0.25),
        /*------------- 4d --------------*/
        BatchNormParams(Shape{2, 2, 2, 1},
                        Shape{2, 2, 2, 1},
                        ET,
                        ET,
                        std::vector<T>{0.54881352f,
                                       0.71518934f,
                                       0.60276335f,
                                       0.54488319f,
                                       0.42365479f,
                                       0.64589411f,
                                       0.4375872f,
                                       0.89177299f},
                        std::vector<T>{0.54903894f,
                                       0.71533161f,
                                       0.60296183f,
                                       0.54511058f,
                                       0.42394274f,
                                       0.64607101f,
                                       0.43786817f,
                                       0.89182704f},
                        std::vector<T>{1.0, 1.0},
                        std::vector<T>{1.0, 1.0},
                        std::vector<T>{1.0, 1.0},
                        std::vector<T>{1.0, 1.0},
                        0.001),
        BatchNormParams(
            Shape{2, 2, 2, 1},
            Shape{2, 2, 2, 1},
            ET,
            ET,
            std::vector<T>{0.54881352f,
                           0.71518934f,
                           0.60276335f,
                           0.54488319f,
                           0.42365479f,
                           0.64589411f,
                           0.4375872f,
                           0.89177299f},
            std::vector<T>{-0.30327f, 1.1561f, -0.096382f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f},
            std::vector<T>{1.0, 1.0},
            std::vector<T>{0.0f, 0.0f},
            std::vector<T>{0.583388f, 0.619252f},
            std::vector<T>{0.0119972f, 0.0282681f},
            0.001),
    };

    return params;
}

std::vector<BatchNormParams> generateCombinedParamsForBatchNorm() {
    const std::vector<std::vector<BatchNormParams>> allTypeParams{generateParamsForBatchNorm<element::Type_t::f32>(),
                                                                  generateParamsForBatchNorm<element::Type_t::f16>()};

    std::vector<BatchNormParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_BatchNorm_With_Hardcoded_Refs,
                         ReferenceBatchNormV0LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForBatchNorm()),
                         ReferenceBatchNormV0LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BatchNorm_With_Hardcoded_Refs,
                         ReferenceBatchNormV5LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForBatchNorm()),
                         ReferenceBatchNormV5LayerTest::getTestCaseName);

}  // namespace

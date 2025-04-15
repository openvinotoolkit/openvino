// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rnn_cell.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct RNNCellParams {
    RNNCellParams(int32_t batchSize,
                  int32_t inputSize,
                  int32_t hiddenSize,
                  const reference_tests::Tensor& X,
                  const reference_tests::Tensor& H_t,
                  const reference_tests::Tensor& W,
                  const reference_tests::Tensor& R,
                  const reference_tests::Tensor& B,
                  const reference_tests::Tensor& Ho,
                  const std::string& testcaseName = "")
        : batchSize(batchSize),
          inputSize(inputSize),
          hiddenSize(hiddenSize),
          X(X),
          H_t(H_t),
          W(W),
          R(R),
          B(B),
          Ho(Ho),
          testcaseName(testcaseName) {}

    int32_t batchSize;
    int32_t inputSize;
    int32_t hiddenSize;
    reference_tests::Tensor X;
    reference_tests::Tensor H_t;
    reference_tests::Tensor W;
    reference_tests::Tensor R;
    reference_tests::Tensor B;
    reference_tests::Tensor Ho;
    std::string testcaseName;
};

class ReferenceRNNCellTest : public testing::TestWithParam<RNNCellParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Ho.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<RNNCellParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bSize=" << param.batchSize;
        result << "_iSize=" << param.inputSize;
        result << "_hSize=" << param.hiddenSize;
        result << "_xType=" << param.X.type;
        result << "_xShape=" << param.X.shape;
        result << "_htType=" << param.H_t.type;
        result << "_htShape=" << param.H_t.shape;
        result << "_wType=" << param.W.type;
        result << "_wShape=" << param.W.shape;
        result << "_rType=" << param.R.type;
        result << "_rShape=" << param.R.shape;
        result << "_hoType=" << param.Ho.type;
        if (param.testcaseName != "") {
            result << "_hoShape=" << param.Ho.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_hoShape=" << param.Ho.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const RNNCellParams& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto rnn_cell = std::make_shared<op::v0::RNNCell>(X, H_t, W, R, B, params.hiddenSize);
        auto function = std::make_shared<Model>(NodeVector{rnn_cell}, ParameterVector{X, H_t, W, R, B});
        return function;
    }
};

class ReferenceRNNCellTestBiasClip : public ReferenceRNNCellTest {
private:
    static std::shared_ptr<Model> CreateFunction(const RNNCellParams& params) {
        float clip = 2.88f;

        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto rnn_cell = std::make_shared<op::v0::RNNCell>(X,
                                                                H_t,
                                                                W,
                                                                R,
                                                                B,
                                                                params.hiddenSize,
                                                                std::vector<std::string>{"tanh"},
                                                                std::vector<float>{},
                                                                std::vector<float>{},
                                                                clip);
        auto function = std::make_shared<Model>(NodeVector{rnn_cell}, ParameterVector{X, H_t, W, R, B});
        return function;
    }
};

class ReferenceRNNCellTestSigmoidActivationFunction : public ReferenceRNNCellTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Ho.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const RNNCellParams& params) {
        float clip = 2.88f;

        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto rnn_cell = std::make_shared<op::v0::RNNCell>(X,
                                                                H_t,
                                                                W,
                                                                R,
                                                                B,
                                                                params.hiddenSize,
                                                                std::vector<std::string>{"sigmoid"},
                                                                std::vector<float>{},
                                                                std::vector<float>{},
                                                                clip);
        auto function = std::make_shared<Model>(NodeVector{rnn_cell}, ParameterVector{X, H_t, W, R, B});
        return function;
    }
};

TEST_P(ReferenceRNNCellTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceRNNCellTestBiasClip, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceRNNCellTestSigmoidActivationFunction, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<RNNCellParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<RNNCellParams> params{
        RNNCellParams(2,
                      3,
                      3,
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f}),
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f}),
                      reference_tests::Tensor(ET,
                                              {3, 3},
                                              std::vector<T>{0.41930267f,
                                                             0.7872176f,
                                                             0.89940447f,
                                                             0.23659843f,
                                                             0.24676207f,
                                                             0.17101714f,
                                                             0.3147149f,
                                                             0.6555601f,
                                                             0.4559603f}),
                      reference_tests::Tensor(ET,
                                              {3, 3},
                                              std::vector<T>{0.8374871f,
                                                             0.86660194f,
                                                             0.82114047f,
                                                             0.71549815f,
                                                             0.18775631f,
                                                             0.3182116f,
                                                             0.25392973f,
                                                             0.38301638f,
                                                             0.85531586f}),
                      reference_tests::Tensor(ET, {3}, std::vector<T>{0.0f, 0.0f, 0.0f}),
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.9408395f, 0.53823817f, 0.84270686f, 0.98932856f, 0.768665f, 0.90461975f}),
                      "rnn_cell_zero_bias_default_attrs"),
    };
    return params;
}

std::vector<RNNCellParams> generateCombinedParams() {
    const std::vector<std::vector<RNNCellParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<RNNCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<RNNCellParams> generateParamsBiasClip() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<RNNCellParams> params{
        RNNCellParams(2,
                      3,
                      3,
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f}),
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f}),
                      reference_tests::Tensor(ET,
                                              {3, 3},
                                              std::vector<T>{0.41930267f,
                                                             0.7872176f,
                                                             0.89940447f,
                                                             0.23659843f,
                                                             0.24676207f,
                                                             0.17101714f,
                                                             0.3147149f,
                                                             0.6555601f,
                                                             0.4559603f}),
                      reference_tests::Tensor(ET,
                                              {3, 3},
                                              std::vector<T>{0.8374871f,
                                                             0.86660194f,
                                                             0.82114047f,
                                                             0.71549815f,
                                                             0.18775631f,
                                                             0.3182116f,
                                                             0.25392973f,
                                                             0.38301638f,
                                                             0.85531586f}),
                      reference_tests::Tensor(ET, {3}, std::vector<T>{1.0289404f, 1.6362579f, 0.4370661f}),
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.9922437f, 0.97749525f, 0.9312212f, 0.9937176f, 0.9901317f, 0.95906746f}),
                      "rnn_cell_bias_clip"),
    };
    return params;
}

std::vector<RNNCellParams> generateCombinedParamsBiasClip() {
    const std::vector<std::vector<RNNCellParams>> generatedParams{
        generateParamsBiasClip<element::Type_t::bf16>(),
        generateParamsBiasClip<element::Type_t::f16>(),
        generateParamsBiasClip<element::Type_t::f32>(),
        generateParamsBiasClip<element::Type_t::f64>(),
    };
    std::vector<RNNCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<RNNCellParams> generateParamsSigmoidActivationFunction() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<RNNCellParams> params{
        RNNCellParams(2,
                      3,
                      3,
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f}),
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f}),
                      reference_tests::Tensor(ET,
                                              {3, 3},
                                              std::vector<T>{0.41930267f,
                                                             0.7872176f,
                                                             0.89940447f,
                                                             0.23659843f,
                                                             0.24676207f,
                                                             0.17101714f,
                                                             0.3147149f,
                                                             0.6555601f,
                                                             0.4559603f}),
                      reference_tests::Tensor(ET,
                                              {3, 3},
                                              std::vector<T>{0.8374871f,
                                                             0.86660194f,
                                                             0.82114047f,
                                                             0.71549815f,
                                                             0.18775631f,
                                                             0.3182116f,
                                                             0.25392973f,
                                                             0.38301638f,
                                                             0.85531586f}),
                      reference_tests::Tensor(ET, {3}, std::vector<T>{1.0289404f, 1.6362579f, 0.4370661f}),
                      reference_tests::Tensor(
                          ET,
                          {2, 3},
                          std::vector<T>{0.94126844f, 0.9036043f, 0.841243f, 0.9468489f, 0.934215f, 0.873708f}),
                      "rnn_cell_sigmoid_activation_function"),
    };
    return params;
}

std::vector<RNNCellParams> generateCombinedParamsSigmoidActivationFunction() {
    const std::vector<std::vector<RNNCellParams>> generatedParams{
        generateParamsSigmoidActivationFunction<element::Type_t::bf16>(),
        generateParamsSigmoidActivationFunction<element::Type_t::f16>(),
        generateParamsSigmoidActivationFunction<element::Type_t::f32>(),
        generateParamsSigmoidActivationFunction<element::Type_t::f64>(),
    };
    std::vector<RNNCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_RNNCell_With_Hardcoded_Refs,
                         ReferenceRNNCellTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceRNNCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RNNCell_With_Hardcoded_Refs,
                         ReferenceRNNCellTestBiasClip,
                         testing::ValuesIn(generateCombinedParamsBiasClip()),
                         ReferenceRNNCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RNNCell_With_Hardcoded_Refs,
                         ReferenceRNNCellTestSigmoidActivationFunction,
                         testing::ValuesIn(generateCombinedParamsSigmoidActivationFunction()),
                         ReferenceRNNCellTest::getTestCaseName);
}  // namespace

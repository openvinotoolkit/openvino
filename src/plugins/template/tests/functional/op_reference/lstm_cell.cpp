// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct LSTMCellParams {
    int32_t batchSize;
    int32_t inputSize;
    int32_t hiddenSize;
    int32_t gatesCount;
    reference_tests::Tensor X;
    reference_tests::Tensor W;
    reference_tests::Tensor R;
    reference_tests::Tensor H_t;
    reference_tests::Tensor C_t;
    reference_tests::Tensor B;
    reference_tests::Tensor P;
    reference_tests::Tensor Ho;
    reference_tests::Tensor Co;
    std::string testcaseName;
};

struct Builder : ParamsBuilder<LSTMCellParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, batchSize);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, inputSize);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, hiddenSize);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, gatesCount);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, X);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, W);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, R);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, H_t);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, C_t);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, B);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, P);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, Ho);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, Co);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, testcaseName);
};

class ReferenceLSTMCellTest : public testing::TestWithParam<LSTMCellParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.C_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Ho.data, params.Co.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<LSTMCellParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bSize=" << param.batchSize;
        result << "_iSize=" << param.inputSize;
        result << "_hSize=" << param.hiddenSize;
        result << "_gCount=" << param.gatesCount;
        result << "_xType=" << param.X.type;
        result << "_xShape=" << param.X.shape;
        result << "_wType=" << param.W.type;
        result << "_wShape=" << param.W.shape;
        result << "_rType=" << param.R.type;
        result << "_rShape=" << param.R.shape;
        result << "_htType=" << param.H_t.type;
        result << "_htShape=" << param.H_t.shape;
        result << "_ctType=" << param.C_t.type;
        result << "_ctShape=" << param.C_t.shape;
        result << "_hoType=" << param.Ho.type;
        result << "_hoShape=" << param.Ho.shape;
        result << "_coType=" << param.Co.type;
        result << "_coShape=" << param.Co.shape;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const LSTMCellParams& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto C_t = std::make_shared<op::v0::Parameter>(params.C_t.type, params.C_t.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto lstm_cell =
            std::make_shared<op::v4::LSTMCell>(X,
                                               H_t,
                                               C_t,
                                               op::util::convert_lstm_node_format(W, op::util::LSTMWeightsFormat::IOFC),
                                               op::util::convert_lstm_node_format(R, op::util::LSTMWeightsFormat::IOFC),
                                               op::util::convert_lstm_node_format(B, op::util::LSTMWeightsFormat::IOFC),
                                               params.hiddenSize);

        auto function = std::make_shared<Model>(lstm_cell->outputs(), ParameterVector{X, H_t, C_t, W, R, B});
        return function;
    }
};

class ReferenceLSTMCellTestBiasDefaultAttrs : public ReferenceLSTMCellTest {
public:
    void SetUp() override {
        legacy_compare = true;
        threshold = 1e-1f;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.C_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Ho.data, params.Co.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const LSTMCellParams& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto C_t = std::make_shared<op::v0::Parameter>(params.C_t.type, params.C_t.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto lstm_cell =
            std::make_shared<op::v4::LSTMCell>(X,
                                               H_t,
                                               C_t,
                                               op::util::convert_lstm_node_format(W, op::util::LSTMWeightsFormat::IOFC),
                                               op::util::convert_lstm_node_format(R, op::util::LSTMWeightsFormat::IOFC),
                                               op::util::convert_lstm_node_format(B, op::util::LSTMWeightsFormat::IOFC),
                                               params.hiddenSize);

        auto function = std::make_shared<Model>(lstm_cell->outputs(), ParameterVector{X, H_t, C_t, W, R, B});
        return function;
    }
};

class ReferenceLSTMCellTestBiasClip : public ReferenceLSTMCellTest {
public:
    void SetUp() override {
        legacy_compare = true;
        threshold = 1e-1f;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.C_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Ho.data, params.Co.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const LSTMCellParams& params) {
        const float clip_threshold = 3.5f;

        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto C_t = std::make_shared<op::v0::Parameter>(params.C_t.type, params.C_t.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto lstm_cell = std::make_shared<op::v4::LSTMCell>(X,
                                                                  H_t,
                                                                  C_t,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  params.hiddenSize,
                                                                  std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                                                  std::vector<float>{},
                                                                  std::vector<float>{},
                                                                  clip_threshold);

        auto function = std::make_shared<Model>(lstm_cell->outputs(), ParameterVector{X, H_t, C_t, W, R, B});
        return function;
    }
};

TEST_P(ReferenceLSTMCellTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceLSTMCellTestBiasDefaultAttrs, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceLSTMCellTestBiasClip, CompareWithRefs) {
    Exec();
}

class ReferenceLSTMCellV1Test : public ReferenceLSTMCellTest {
private:
    static std::shared_ptr<Model> CreateFunction(const LSTMCellParams& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto C_t = std::make_shared<op::v0::Parameter>(params.C_t.type, params.C_t.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto lstm_cell =
            std::make_shared<op::v0::LSTMCell>(X,
                                               H_t,
                                               C_t,
                                               op::util::convert_lstm_node_format(W, op::util::LSTMWeightsFormat::IOFC),
                                               op::util::convert_lstm_node_format(R, op::util::LSTMWeightsFormat::IOFC),
                                               op::util::convert_lstm_node_format(B, op::util::LSTMWeightsFormat::IOFC),
                                               params.hiddenSize);

        auto function = std::make_shared<Model>(lstm_cell->outputs(), ParameterVector{X, H_t, C_t, W, R, B});
        return function;
    }
};

class ReferenceLSTMCellV1TestBiasDefaultAttrs : public ReferenceLSTMCellTestBiasDefaultAttrs {
private:
    static std::shared_ptr<Model> CreateFunction(const LSTMCellParams& params) {
        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto C_t = std::make_shared<op::v0::Parameter>(params.C_t.type, params.C_t.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto lstm_cell =
            std::make_shared<op::v0::LSTMCell>(X,
                                               H_t,
                                               C_t,
                                               op::util::convert_lstm_node_format(W, op::util::LSTMWeightsFormat::IOFC),
                                               op::util::convert_lstm_node_format(R, op::util::LSTMWeightsFormat::IOFC),
                                               op::util::convert_lstm_node_format(B, op::util::LSTMWeightsFormat::IOFC),
                                               params.hiddenSize);

        auto function = std::make_shared<Model>(lstm_cell->outputs(), ParameterVector{X, H_t, C_t, W, R, B});
        return function;
    }
};

class ReferenceLSTMCellV1TestBiasClip : public ReferenceLSTMCellTestBiasClip {
public:
    void SetUp() override {
        legacy_compare = true;
        threshold = 1e-1f;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data,
                     params.H_t.data,
                     params.C_t.data,
                     params.W.data,
                     params.R.data,
                     params.B.data,
                     params.P.data};
        refOutData = {params.Ho.data, params.Co.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const LSTMCellParams& params) {
        const float clip_threshold = 3.5f;

        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto C_t = std::make_shared<op::v0::Parameter>(params.C_t.type, params.C_t.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto P = std::make_shared<op::v0::Parameter>(params.P.type, params.P.shape);

        const auto lstm_cell = std::make_shared<op::v0::LSTMCell>(X,
                                                                  H_t,
                                                                  C_t,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  P,
                                                                  params.hiddenSize,
                                                                  op::LSTMWeightsFormat::FICO,
                                                                  std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                                                  std::vector<float>{},
                                                                  std::vector<float>{},
                                                                  clip_threshold,
                                                                  false);

        auto function = std::make_shared<Model>(lstm_cell->outputs(), ParameterVector{X, H_t, C_t, W, R, B, P});
        return function;
    }
};

TEST_P(ReferenceLSTMCellV1Test, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceLSTMCellV1TestBiasDefaultAttrs, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceLSTMCellV1TestBiasClip, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<LSTMCellParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<LSTMCellParams> params{
        Builder{}
            .batchSize(2)
            .inputSize(3)
            .hiddenSize(3)
            .gatesCount(4)
            .X(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f}))
            .W(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{
                    3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f, 7.3950343e-02f, 3.8063636e-01f,
                    9.6921772e-01f, 9.6897459e-01f, 6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                    6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f, 4.0472135e-01f, 6.8342745e-01f,
                    8.3432144e-01f, 4.4928190e-01f, 7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                    5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f, 2.1716513e-01f, 2.7473119e-01f,
                    3.3999152e-02f, 9.6835363e-01f, 3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f}))
            .R(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
                               0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
                               0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
                               0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
                               0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
                               0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f}))
            .H_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f}))
            .C_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f}))
            .B(reference_tests::Tensor(ET, {4 * 3}, std::vector<T>(4 * 3, 0.f)))
            .P(reference_tests::Tensor(ET, {3 * 3}, std::vector<T>(3 * 3, 0.f)))
            .Ho(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.81457126f, 0.61109227f, 0.769522f, 0.52239674f, 0.4324641f, 0.63183f}))
            .Co(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{1.4444952f, 0.9635685f, 1.2875274f, 0.8053419f, 0.7184521f, 0.95803297f}))
            .testcaseName("lstm_cell_zero_bias_default_attrs")};
    return params;
}

std::vector<LSTMCellParams> generateCombinedParams() {
    const std::vector<std::vector<LSTMCellParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<LSTMCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<LSTMCellParams> generateParamsBiasDefaultAttrs() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<LSTMCellParams> params{
        Builder{}
            .batchSize(2)
            .inputSize(3)
            .hiddenSize(3)
            .gatesCount(4)
            .X(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f}))
            .W(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{
                    3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f, 7.3950343e-02f, 3.8063636e-01f,
                    9.6921772e-01f, 9.6897459e-01f, 6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                    6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f, 4.0472135e-01f, 6.8342745e-01f,
                    8.3432144e-01f, 4.4928190e-01f, 7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                    5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f, 2.1716513e-01f, 2.7473119e-01f,
                    3.3999152e-02f, 9.6835363e-01f, 3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f}))
            .R(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
                               0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
                               0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
                               0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
                               0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
                               0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f}))
            .H_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f}))
            .C_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f}))
            .B(reference_tests::Tensor(ET,
                                       {4 * 3},
                                       std::vector<T>{1.07393714f,
                                                      1.15248052f,
                                                      1.16671345f,
                                                      0.21450312f,
                                                      1.2380678f,
                                                      1.51688835f,
                                                      0.46718366f,
                                                      0.91810346f,
                                                      1.1274234f,
                                                      0.51022074f,
                                                      1.11389844f,
                                                      0.74174305f}))
            .P(reference_tests::Tensor(ET, {3 * 3}, std::vector<T>(3 * 3, 0.f)))
            .Ho(reference_tests::Tensor(ET,
                                        {2, 3},
                                        std::vector<T>{0.81014400720596313,
                                                       0.76665538549423218,
                                                       0.82509011030197144,
                                                       0.6479143500328064,
                                                       0.66586339473724365,
                                                       0.74838578701019287}))
            .Co(reference_tests::Tensor(ET,
                                        {2, 3},
                                        std::vector<T>{1.6800162792205811,
                                                       1.1150213479995728,
                                                       1.4578367471694946,
                                                       1.0649888515472412,
                                                       0.93761754035949707,
                                                       1.3659683465957642}))
            .testcaseName("lstm_cell_bias_default_attrs"),
    };
    return params;
}

std::vector<LSTMCellParams> generateCombinedParamsBiasDefaultAttrs() {
    const std::vector<std::vector<LSTMCellParams>> generatedParams{
        generateParamsBiasDefaultAttrs<element::Type_t::bf16>(),
        generateParamsBiasDefaultAttrs<element::Type_t::f16>(),
        generateParamsBiasDefaultAttrs<element::Type_t::f32>(),
        generateParamsBiasDefaultAttrs<element::Type_t::f64>(),
    };
    std::vector<LSTMCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<LSTMCellParams> generateParamsBiasClip() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<LSTMCellParams> params{
        Builder{}
            .batchSize(2)
            .inputSize(3)
            .hiddenSize(3)
            .gatesCount(4)
            .X(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f}))
            .W(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{
                    3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f, 7.3950343e-02f, 3.8063636e-01f,
                    9.6921772e-01f, 9.6897459e-01f, 6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                    6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f, 4.0472135e-01f, 6.8342745e-01f,
                    8.3432144e-01f, 4.4928190e-01f, 7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                    5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f, 2.1716513e-01f, 2.7473119e-01f,
                    3.3999152e-02f, 9.6835363e-01f, 3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f}))
            .R(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
                               0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
                               0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
                               0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
                               0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
                               0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f}))
            .H_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f}))
            .C_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f}))
            .B(reference_tests::Tensor(ET,
                                       {4 * 3},
                                       std::vector<T>{1.07393714f,
                                                      1.15248052f,
                                                      1.16671345f,
                                                      0.21450312f,
                                                      1.2380678f,
                                                      1.51688835f,
                                                      0.46718366f,
                                                      0.91810346f,
                                                      1.1274234f,
                                                      0.51022074f,
                                                      1.11389844f,
                                                      0.74174305f}))
            .P(reference_tests::Tensor(ET, {3 * 3}, std::vector<T>(3 * 3, 0.f)))
            .Ho(reference_tests::Tensor(ET,
                                        {2, 3},
                                        std::vector<T>{0.81014400720596313,
                                                       0.76665538549423218,
                                                       0.82387429475784302,
                                                       0.6479143500328064,
                                                       0.66586339473724365,
                                                       0.74838578701019287}))
            .Co(reference_tests::Tensor(ET,
                                        {2, 3},
                                        std::vector<T>{1.6800162792205811,
                                                       1.1150213479995728,
                                                       1.4510968923568726,
                                                       1.0649888515472412,
                                                       0.93761754035949707,
                                                       1.3659683465957642}))
            .testcaseName("lstm_cell_bias_clip"),
    };
    return params;
}

std::vector<LSTMCellParams> generateCombinedParamsBiasClip() {
    const std::vector<std::vector<LSTMCellParams>> generatedParams{
        generateParamsBiasClip<element::Type_t::bf16>(),
        generateParamsBiasClip<element::Type_t::f16>(),
        generateParamsBiasClip<element::Type_t::f32>(),
        generateParamsBiasClip<element::Type_t::f64>(),
    };
    std::vector<LSTMCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCell_With_Hardcoded_Refs,
                         ReferenceLSTMCellTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceLSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCell_With_Hardcoded_Refs,
                         ReferenceLSTMCellTestBiasDefaultAttrs,
                         testing::ValuesIn(generateCombinedParamsBiasDefaultAttrs()),
                         ReferenceLSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCell_With_Hardcoded_Refs,
                         ReferenceLSTMCellTestBiasClip,
                         testing::ValuesIn(generateCombinedParamsBiasClip()),
                         ReferenceLSTMCellTest::getTestCaseName);

template <element::Type_t ET>
std::vector<LSTMCellParams> generateParamsV1() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<LSTMCellParams> params{
        Builder{}
            .batchSize(2)
            .inputSize(3)
            .hiddenSize(3)
            .gatesCount(4)
            .X(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f}))
            .W(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{
                    3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f, 7.3950343e-02f, 3.8063636e-01f,
                    9.6921772e-01f, 9.6897459e-01f, 6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                    6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f, 4.0472135e-01f, 6.8342745e-01f,
                    8.3432144e-01f, 4.4928190e-01f, 7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                    5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f, 2.1716513e-01f, 2.7473119e-01f,
                    3.3999152e-02f, 9.6835363e-01f, 3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f}))
            .R(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
                               0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
                               0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
                               0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
                               0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
                               0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f}))
            .H_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f}))
            .C_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f}))
            .B(reference_tests::Tensor(ET, {4 * 3}, std::vector<T>(4 * 3, 0.f)))
            .P(reference_tests::Tensor(ET, {3 * 3}, std::vector<T>(3 * 3, 0.f)))
            .Ho(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.81457126f, 0.61109227f, 0.769522f, 0.52239674f, 0.4324641f, 0.63183f}))
            .Co(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{1.4444952f, 0.9635685f, 1.2875274f, 0.8053419f, 0.7184521f, 0.95803297f}))
            .testcaseName("lstm_cell_v1_zero_bias_default_attrs")};
    return params;
}

std::vector<LSTMCellParams> generateCombinedParamsV1() {
    const std::vector<std::vector<LSTMCellParams>> generatedParams{
        generateParamsV1<element::Type_t::bf16>(),
        generateParamsV1<element::Type_t::f16>(),
        generateParamsV1<element::Type_t::f32>(),
        generateParamsV1<element::Type_t::f64>(),
    };
    std::vector<LSTMCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<LSTMCellParams> generateParamsBiasDefaultAttrsV1() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<LSTMCellParams> params{
        Builder{}
            .batchSize(2)
            .inputSize(3)
            .hiddenSize(3)
            .gatesCount(4)
            .X(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f}))
            .W(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{
                    3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f, 7.3950343e-02f, 3.8063636e-01f,
                    9.6921772e-01f, 9.6897459e-01f, 6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                    6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f, 4.0472135e-01f, 6.8342745e-01f,
                    8.3432144e-01f, 4.4928190e-01f, 7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                    5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f, 2.1716513e-01f, 2.7473119e-01f,
                    3.3999152e-02f, 9.6835363e-01f, 3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f}))
            .R(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
                               0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
                               0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
                               0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
                               0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
                               0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f}))
            .H_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f}))
            .C_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f}))
            .B(reference_tests::Tensor(ET,
                                       {4 * 3},
                                       std::vector<T>{1.07393714f,
                                                      1.15248052f,
                                                      1.16671345f,
                                                      0.21450312f,
                                                      1.2380678f,
                                                      1.51688835f,
                                                      0.46718366f,
                                                      0.91810346f,
                                                      1.1274234f,
                                                      0.51022074f,
                                                      1.11389844f,
                                                      0.74174305f}))
            .P(reference_tests::Tensor(ET, {3 * 3}, std::vector<T>(3 * 3, 0.f)))
            .Ho(reference_tests::Tensor(ET,
                                        {2, 3},
                                        std::vector<T>{0.81014400720596313,
                                                       0.76665538549423218,
                                                       0.82509011030197144,
                                                       0.6479143500328064,
                                                       0.66586339473724365,
                                                       0.74838578701019287}))
            .Co(reference_tests::Tensor(ET,
                                        {2, 3},
                                        std::vector<T>{1.6800162792205811,
                                                       1.1150213479995728,
                                                       1.4578367471694946,
                                                       1.0649888515472412,
                                                       0.93761754035949707,
                                                       1.3659683465957642}))
            .testcaseName("lstm_cell_v1_bias_default_attrs"),
    };
    return params;
}

std::vector<LSTMCellParams> generateCombinedParamsBiasDefaultAttrsV1() {
    const std::vector<std::vector<LSTMCellParams>> generatedParams{
        generateParamsBiasDefaultAttrsV1<element::Type_t::bf16>(),
        generateParamsBiasDefaultAttrsV1<element::Type_t::f16>(),
        generateParamsBiasDefaultAttrsV1<element::Type_t::f32>(),
        generateParamsBiasDefaultAttrsV1<element::Type_t::f64>(),
    };
    std::vector<LSTMCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<LSTMCellParams> generateParamsBiasClipV1() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<LSTMCellParams> params{
        Builder{}
            .batchSize(2)
            .inputSize(3)
            .hiddenSize(3)
            .gatesCount(4)
            .X(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f}))
            .W(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{
                    3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f, 7.3950343e-02f, 3.8063636e-01f,
                    9.6921772e-01f, 9.6897459e-01f, 6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                    6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f, 4.0472135e-01f, 6.8342745e-01f,
                    8.3432144e-01f, 4.4928190e-01f, 7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                    5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f, 2.1716513e-01f, 2.7473119e-01f,
                    3.3999152e-02f, 9.6835363e-01f, 3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f}))
            .R(reference_tests::Tensor(
                ET,
                {4 * 3, 3},
                std::vector<T>{0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
                               0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
                               0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
                               0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
                               0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
                               0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f}))
            .H_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f}))
            .C_t(reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f}))
            .B(reference_tests::Tensor(ET,
                                       {4 * 3},
                                       std::vector<T>{1.07393714f,
                                                      1.15248052f,
                                                      1.16671345f,
                                                      0.21450312f,
                                                      1.2380678f,
                                                      1.51688835f,
                                                      0.46718366f,
                                                      0.91810346f,
                                                      1.1274234f,
                                                      0.51022074f,
                                                      1.11389844f,
                                                      0.74174305f}))
            .P(reference_tests::Tensor(ET, {3 * 3}, std::vector<T>(3 * 3, 0.f)))
            .Ho(reference_tests::Tensor(ET,
                                        {2, 3},
                                        std::vector<T>{0.81014400720596313,
                                                       0.76665538549423218,
                                                       0.82387429475784302,
                                                       0.6479143500328064,
                                                       0.66586339473724365,
                                                       0.74838578701019287}))
            .Co(reference_tests::Tensor(ET,
                                        {2, 3},
                                        std::vector<T>{1.6800162792205811,
                                                       1.1150213479995728,
                                                       1.4510968923568726,
                                                       1.0649888515472412,
                                                       0.93761754035949707,
                                                       1.3659683465957642}))
            .testcaseName("lstm_cell_v1_bias_clip"),
    };
    return params;
}

std::vector<LSTMCellParams> generateCombinedParamsBiasClipV1() {
    const std::vector<std::vector<LSTMCellParams>> generatedParams{
        generateParamsBiasClipV1<element::Type_t::bf16>(),
        generateParamsBiasClipV1<element::Type_t::f16>(),
        generateParamsBiasClipV1<element::Type_t::f32>(),
        generateParamsBiasClipV1<element::Type_t::f64>(),
    };
    std::vector<LSTMCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellV1_With_Hardcoded_Refs,
                         ReferenceLSTMCellV1Test,
                         testing::ValuesIn(generateCombinedParamsV1()),
                         ReferenceLSTMCellV1Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellV1_With_Hardcoded_Refs,
                         ReferenceLSTMCellV1TestBiasDefaultAttrs,
                         testing::ValuesIn(generateCombinedParamsBiasDefaultAttrsV1()),
                         ReferenceLSTMCellV1Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellV1_With_Hardcoded_Refs,
                         ReferenceLSTMCellV1TestBiasClip,
                         testing::ValuesIn(generateCombinedParamsBiasClipV1()),
                         ReferenceLSTMCellV1Test::getTestCaseName);
}  // namespace

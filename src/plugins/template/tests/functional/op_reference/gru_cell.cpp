// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gru_cell.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GRUCellParams {
    GRUCellParams(const int32_t batchSize,
                  const int32_t inputSize,
                  const int32_t hiddenSize,
                  const int32_t gatesCount,
                  const bool linearBeforeReset,
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
          gatesCount(gatesCount),
          linearBeforeReset(linearBeforeReset),
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
    int32_t gatesCount;
    bool linearBeforeReset;
    reference_tests::Tensor X;
    reference_tests::Tensor H_t;
    reference_tests::Tensor W;
    reference_tests::Tensor R;
    reference_tests::Tensor B;
    reference_tests::Tensor Ho;
    std::string testcaseName;
};

class ReferenceGRUCellTest : public testing::TestWithParam<GRUCellParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Ho.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GRUCellParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bSize=" << param.batchSize;
        result << "_iSize=" << param.inputSize;
        result << "_hSize=" << param.hiddenSize;
        result << "_gCount=" << param.gatesCount;
        result << "_lbReset=" << param.linearBeforeReset;
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
    static std::shared_ptr<Model> CreateFunction(const GRUCellParams& params) {
        float clip = 2.88f;

        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto gru_cell = std::make_shared<op::v3::GRUCell>(X,
                                                                H_t,
                                                                W,
                                                                R,
                                                                B,
                                                                params.hiddenSize,
                                                                std::vector<std::string>{"sigmoid", "tanh"},
                                                                std::vector<float>{},
                                                                std::vector<float>{},
                                                                clip,
                                                                params.linearBeforeReset);

        auto function = std::make_shared<Model>(NodeVector{gru_cell}, ParameterVector{X, H_t, W, R, B});
        return function;
    }
};

// Hard Sigmoid activation function is unsupprted with v3::GRUCell
class ReferenceGRUCellTestHardsigmoidActivationFunction : public ReferenceGRUCellTest {
public:
    void SetUp() override {
        threshold = 1e-1f;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.X.data, params.H_t.data, params.W.data, params.R.data, params.B.data};
        refOutData = {params.Ho.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const GRUCellParams& params) {
        float clip = 2.88f;

        const auto X = std::make_shared<op::v0::Parameter>(params.X.type, params.X.shape);
        const auto H_t = std::make_shared<op::v0::Parameter>(params.H_t.type, params.H_t.shape);
        const auto W = std::make_shared<op::v0::Parameter>(params.W.type, params.W.shape);
        const auto R = std::make_shared<op::v0::Parameter>(params.R.type, params.R.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);

        const auto gru_cell = std::make_shared<op::v3::GRUCell>(X,
                                                                H_t,
                                                                W,
                                                                R,
                                                                B,
                                                                params.hiddenSize,
                                                                std::vector<std::string>{"hardsigmoid", "hardsigmoid"},
                                                                std::vector<float>{1.8345f, 1.8345f},
                                                                std::vector<float>{3.05f, 3.05f},
                                                                clip,
                                                                params.linearBeforeReset);

        auto function = std::make_shared<Model>(NodeVector{gru_cell}, ParameterVector{X, H_t, W, R, B});
        return function;
    }
};

TEST_P(ReferenceGRUCellTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceGRUCellTestHardsigmoidActivationFunction, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<GRUCellParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<GRUCellParams> params{
        GRUCellParams(
            2,
            3,
            3,
            3,
            false,
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.52421564f, 0.78845507f, 0.9372873f, 0.59783894f, 0.18278378f, 0.2084126f}),
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.45738035f, 0.996877f, 0.82882977f, 0.47492632f, 0.88471466f, 0.57833236f}),
            reference_tests::Tensor(
                ET,
                {3 * 3, 3},
                std::vector<T>{0.5815369f, 0.16559383f, 0.08464007f, 0.843122f,   0.73968244f, 0.11359601f, 0.8295078f,
                               0.9240567f, 0.10007995f, 0.20573162f, 0.09002485f, 0.2839569f,  0.3096991f,  0.5638341f,
                               0.5787327f, 0.84552664f, 0.16263747f, 0.7243242f,  0.8049057f,  0.43966424f, 0.46294412f,
                               0.9833361f, 0.31369713f, 0.1719934f,  0.4937093f,  0.6353004f,  0.77982515f}),
            reference_tests::Tensor(ET, {3 * 3, 3}, std::vector<T>{0.16510165f, 0.52435565f, 0.2788478f,  0.99427545f,
                                                                   0.1623331f,  0.01389796f, 0.99669236f, 0.53901845f,
                                                                   0.8737506f,  0.9254788f,  0.21172932f, 0.11634306f,
                                                                   0.40111724f, 0.37497616f, 0.2903471f,  0.6796794f,
                                                                   0.65131867f, 0.78163475f, 0.12058706f, 0.45591718f,
                                                                   0.791677f,   0.76497287f, 0.9895242f,  0.7845312f,
                                                                   0.51267904f, 0.49030215f, 0.08498167f}),
            reference_tests::Tensor(ET,
                                    {3 * 3},
                                    std::vector<T>{0.8286678f + 0.9175602f,
                                                   0.9153158f + 0.14958014f,
                                                   0.9581612f + 0.49230585f,
                                                   0.6639213f + 0.63162816f,
                                                   0.84239805f + 0.4161903f,
                                                   0.5282445f + 0.22148274f,
                                                   0.14153397f + 0.50496656f,
                                                   0.22404431f + 0.34798595f,
                                                   0.6549655f + 0.6699164f}),
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.48588726f, 0.99670005f, 0.83759373f, 0.5023099f, 0.89410484f, 0.60011315f}),
            "gru_cell_bias_clip"),
        GRUCellParams(
            2,
            3,
            3,
            3,
            true,
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f}),
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f}),
            reference_tests::Tensor(ET, {3 * 3, 3}, std::vector<T>{0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f,
                                                                   0.6964006f,  0.33459795f, 0.5468904f,  0.85646594f,
                                                                   0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f,
                                                                   0.56943774f, 0.7475505f,  0.2490578f,  0.86977345f,
                                                                   0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,
                                                                   0.53454477f, 0.15974349f, 0.5804805f,  0.14303213f,
                                                                   0.07514781f, 0.5865731f,  0.76409274f}),
            reference_tests::Tensor(ET, {3 * 3, 3}, std::vector<T>{0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f,
                                                                   0.04471736f, 0.03888785f, 0.06308217f, 0.44844428f,
                                                                   0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,
                                                                   0.63143945f, 0.00277612f, 0.37198433f, 0.06966069f,
                                                                   0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f,
                                                                   0.10751773f, 0.18332677f, 0.1326976f,  0.9998985f,
                                                                   0.19263928f, 0.10979804f, 0.52575564f}),
            reference_tests::Tensor(ET,
                                    {(3 + 1) * 3},
                                    std::vector<T>{0.61395123f,  // 0.09875853f + 0.5151927f,
                                                   1.08667738f,  // 0.37801138f + 0.708666f,
                                                   1.32600244f,  // 0.7729636f + 0.55303884f,
                                                   0.81917698f,  // 0.78493553f + 0.03424145f,
                                                   1.37736335f,  // 0.5662702f + 0.81109315f,
                                                   0.42931147f,  // 0.12406381f + 0.30524766f,
                                                   0.66729516f,
                                                   0.7752771f,
                                                   0.78819966f,
                                                   0.6606634f,
                                                   0.99040645f,
                                                   0.21112025f}),
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8709214f, 0.48411977f, 0.74495184f, 0.6074972f, 0.44572943f, 0.1467715f}),
            "gru_cell_linear_before_reset"),
    };
    return params;
}

std::vector<GRUCellParams> generateCombinedParams() {
    const std::vector<std::vector<GRUCellParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<GRUCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<GRUCellParams> generateParamsHardsigmoidActivationFunction() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<GRUCellParams> params{
        GRUCellParams(
            2,
            3,
            3,
            3,
            true,
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f}),
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f}),
            reference_tests::Tensor(ET, {3 * 3, 3}, std::vector<T>{0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f,
                                                                   0.6964006f,  0.33459795f, 0.5468904f,  0.85646594f,
                                                                   0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f,
                                                                   0.56943774f, 0.7475505f,  0.2490578f,  0.86977345f,
                                                                   0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,
                                                                   0.53454477f, 0.15974349f, 0.5804805f,  0.14303213f,
                                                                   0.07514781f, 0.5865731f,  0.76409274f}),
            reference_tests::Tensor(ET, {3 * 3, 3}, std::vector<T>{0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f,
                                                                   0.04471736f, 0.03888785f, 0.06308217f, 0.44844428f,
                                                                   0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,
                                                                   0.63143945f, 0.00277612f, 0.37198433f, 0.06966069f,
                                                                   0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f,
                                                                   0.10751773f, 0.18332677f, 0.1326976f,  0.9998985f,
                                                                   0.19263928f, 0.10979804f, 0.52575564f}),
            reference_tests::Tensor(ET,
                                    {(3 + 1) * 3},
                                    std::vector<T>{0.09875853f + 0.5151927f,
                                                   0.37801138f + 0.708666f,
                                                   0.7729636f + 0.55303884f,
                                                   0.78493553f + 0.03424145f,
                                                   0.5662702f + 0.81109315f,
                                                   0.12406381f + 0.30524766f,
                                                   0.66729516f,
                                                   0.7752771f,
                                                   0.78819966f,
                                                   0.6606634f,
                                                   0.99040645f,
                                                   0.21112025f}),
            reference_tests::Tensor(
                ET,
                {2, 3},
                std::vector<T>{0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f}),
            "gru_cell_hardsigmoid_activation_function"),
    };
    return params;
}

std::vector<GRUCellParams> generateCombinedParamsHardsigmoidActivationFunction() {
    const std::vector<std::vector<GRUCellParams>> generatedParams{
        generateParamsHardsigmoidActivationFunction<element::Type_t::bf16>(),
        generateParamsHardsigmoidActivationFunction<element::Type_t::f16>(),
        generateParamsHardsigmoidActivationFunction<element::Type_t::f32>(),
        generateParamsHardsigmoidActivationFunction<element::Type_t::f64>(),
    };
    std::vector<GRUCellParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GRUCell_With_Hardcoded_Refs,
                         ReferenceGRUCellTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceGRUCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GRUCell_With_Hardcoded_Refs,
                         ReferenceGRUCellTestHardsigmoidActivationFunction,
                         testing::ValuesIn(generateCombinedParamsHardsigmoidActivationFunction()),
                         ReferenceGRUCellTest::getTestCaseName);
}  // namespace

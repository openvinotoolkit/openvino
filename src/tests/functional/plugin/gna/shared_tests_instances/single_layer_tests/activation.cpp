// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {

class ActivationLayerGNATest : public ActivationLayerTest {
protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationDecl;
        std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, targetDevice) = GetParam();

        auto inputShape = shapes.first;
        const std::size_t inputDim = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<size_t>());
        std::vector<size_t> inputDims{1, inputDim};
        activationType = activationDecl.first;
        auto constantsValue = activationDecl.second;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputDims});
        params[0]->set_friendly_name("Input");

        // TODO: remove after integer inference output support
        if (ngPrc == ngraph::element::u8 || ngPrc == ngraph::element::i16) {
            threshold = 1.0;
        }

        auto inputReshapePattern = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{inputShape.size()}, inputShape);
        auto inputReshape = std::make_shared<ngraph::opset1::Reshape>(params[0], inputReshapePattern, false);
        auto activation = ngraph::builder::makeActivation(inputReshape, ngPrc, activationType, shapes.second, constantsValue);
        auto outputReshapePattern = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2}, inputDims);
        auto outputReshape = std::make_shared<ngraph::opset1::Reshape>(activation, outputReshapePattern, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{outputReshape}, params);
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        bool inPrcSigned = function->get_parameters()[0]->get_element_type().is_signed();
        bool inPrcFloat = info.getPrecision().is_float();
        int32_t data_start_from = -10;
        uint32_t data_range = 20;
        int32_t resolution = 32768;

        switch (activationType) {
            case ngraph::helpers::ActivationTypes::Log: {
                data_start_from = 1;
                data_range = 20;
                resolution = 32768;
                break;
            }
            case ngraph::helpers::ActivationTypes::Sign: {
                data_start_from = -10;
                data_range = 20;
                resolution = 3072;
                break;
            }
            case ngraph::helpers::ActivationTypes::Exp: {
                const double max_result_on_GNA = 15.9;
                const double exp_inverse = std::round(std::log(max_result_on_GNA));
                if (inPrcSigned) {
                    data_range = exp_inverse * 2.0;
                    data_start_from = -exp_inverse;
                } else {
                    data_range = exp_inverse;
                    data_start_from = 0;
                }
                break;
            }
        }
        if (!inPrcSigned) {
            data_range = 15;
            data_start_from = 0;
        }
        if (!inPrcFloat) {
            resolution = 1;
        }

        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_range,
                                                data_start_from, resolution);
    }
};

TEST_P(ActivationLayerGNATest, CompareWithRefs) {
        Run();
}

}  //  namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {
// Common params

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8,
};

const std::vector<InferenceEngine::Precision> preluNetPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid,  {}},
        {Tanh,     {}},
        {Relu,     {}},
        {Exp,      {}},
        {Log,      {}},
        {Sign,     {}},
        {Abs,      {}},
        {Clamp,    {{-5, 5}}}
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> preluActivationTypes = {
        {PReLu,   {{-0.01f}}},
        {LeakyRelu, {{0.01f}}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
        {{1, 10 * 1024}, {{}}},
        {{8, 128}, {{}}},
        {{1, 4, 2, 256}, {{}}},
        {{4, 4, 4, 4}, {{}}},
        {{1, 16, 1, 128}, {{}}},
        {{1, 8, 15, 128}, {{}}},
        {{1, 4, 4, 128}, {{}}},
        {{8}, {{}}},
        {{5}, {{}}},
        {{1, 936, 513}, {{}}},
        {{2, 32, 8}, {{}}}
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_GNA)
);

const auto preluCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(preluActivationTypes)),
        ::testing::ValuesIn(preluNetPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_GNA)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationLayerGNATest, basicCases, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu, ActivationLayerGNATest, preluCases, ActivationLayerTest::getTestCaseName);

}  // namespace

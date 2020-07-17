// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/activation.hpp"

namespace LayerTestsDefinitions {

std::string ActivationLayerTest::getTestCaseName(const testing::TestParamInfo<activationParams> &obj) {
    InferenceEngine::Precision netPrecision;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::string targetDevice;
    ngraph::helpers::ActivationTypes activationType;
    std::tie(activationType, netPrecision, shapes, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << activationNames[activationType] << separator;
    result << "IS=" << CommonTestUtils::vec2str(shapes.first) << separator;;
    result << "AS=" << CommonTestUtils::vec2str(shapes.second) << separator;;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ActivationLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::tie(activationType, netPrecision, shapes, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {shapes.first});
    auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType, shapes.second);
    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params);
}

InferenceEngine::Blob::Ptr ActivationLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    bool inPrcSigned = function->get_parameters()[0]->get_element_type().is_signed();
    int32_t data_start_from;
    uint32_t data_range;

    switch (activationType) {
        case ngraph::helpers::ActivationTypes::Log: {
            data_start_from = 1;
            data_range = 20;
            break;
        }
        case ngraph::helpers::ActivationTypes::Sqrt: {
            data_start_from = 0;
            data_range = 20;
            break;
        }
        case ngraph::helpers::ActivationTypes::Asin: {
            data_start_from = -1;
            data_range = 2;
            break;
        }
        case ngraph::helpers::ActivationTypes::Acos: {
            data_start_from = -1;
            data_range = 2;
            break;
        }
        default: {
            data_start_from = -10;
            data_range = 20;
            break;
        }
    }
    if (!inPrcSigned) {
        data_range = 15;
        data_start_from = 0;
    }
    if (activationType == ngraph::helpers::ActivationTypes::Exp && targetDevice == CommonTestUtils::DEVICE_GNA) {
        const double max_result_on_GNA = 15.9;
        const double exp_inverse = std::round(std::log(max_result_on_GNA));
        if (inPrcSigned) {
            data_range = exp_inverse * 2.0;
            data_start_from = -exp_inverse;
        } else {
            data_range = exp_inverse;
            data_start_from = 0;
        }
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_range,
                                            data_start_from,
                                            32768);
}

ngraph::ParameterVector ActivationParamLayerTest::createActivationParams(ngraph::element::Type ngPrc, std::vector<size_t> inShape) {
    switch (activationType) {
        case ngraph::helpers::ActivationTypes::PReLu: {
            auto negativeSlopeParam = ngraph::builder::makeParams(ngPrc, {inShape});
            negativeSlopeParam[0]->set_friendly_name("negativeSlope");
            return negativeSlopeParam;
        }
        case ngraph::helpers::ActivationTypes::LeakyRelu: {
            auto leakySlopeParam = ngraph::builder::makeParams(ngPrc, {inShape});
            leakySlopeParam[0]->set_friendly_name("leakySlope");
            return leakySlopeParam;
        }
        case ngraph::helpers::ActivationTypes::HardSigmoid: {
            auto hardSigmoidParam = ngraph::builder::makeParams(ngPrc, {inShape, inShape});
            hardSigmoidParam[0]->set_friendly_name("alpha");
            hardSigmoidParam[1]->set_friendly_name("beta");
            return hardSigmoidParam;
        }
        case ngraph::helpers::ActivationTypes::Selu: {
            auto seluParam = ngraph::builder::makeParams(ngPrc, {inShape, inShape});
            seluParam[0]->set_friendly_name("alpha");
            seluParam[1]->set_friendly_name("lambda");
            return seluParam;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported activation type for Params test type";
    }
}

void ActivationParamLayerTest::generateActivationBlob() {
    switch (activationType) {
        case ngraph::helpers::ActivationTypes::PReLu: {
            auto blobNegativeSlope = inferRequest.GetBlob("negativeSlope");
            float negativeSlope = -0.01f;
            blobNegativeSlope = FuncTestUtils::createAndFillBlobWithFloatArray(blobNegativeSlope->getTensorDesc(), &negativeSlope, 1);
        }
        case ngraph::helpers::ActivationTypes::LeakyRelu: {
            auto blobLeakySlope = inferRequest.GetBlob("leakySlope");
            float leakySlope = 0.01f;
            blobLeakySlope = FuncTestUtils::createAndFillBlobWithFloatArray(blobLeakySlope->getTensorDesc(), &leakySlope, 1);
        }
        case ngraph::helpers::ActivationTypes::HardSigmoid: {
            auto blobHardSigmoidAlpha = inferRequest.GetBlob("alpha");
            auto blobHardSigmoidBeta = inferRequest.GetBlob("beta");
            float alpha = 0.2f, beta = 0.5f;
            blobHardSigmoidAlpha = FuncTestUtils::createAndFillBlobWithFloatArray(blobHardSigmoidAlpha->getTensorDesc(), &alpha, 1);
            blobHardSigmoidBeta = FuncTestUtils::createAndFillBlobWithFloatArray(blobHardSigmoidBeta->getTensorDesc(), &beta, 1);
        }
        case ngraph::helpers::ActivationTypes::Selu: {
            auto blobHardSigmoidAlpha = inferRequest.GetBlob("alpha");
            auto blobHardSigmoidLambda = inferRequest.GetBlob("lambda");
            float alpha = 1.6732f, lambda = 1.0507f;
            blobHardSigmoidAlpha = FuncTestUtils::createAndFillBlobWithFloatArray(blobHardSigmoidAlpha->getTensorDesc(), &alpha, 1);
            blobHardSigmoidLambda = FuncTestUtils::createAndFillBlobWithFloatArray(blobHardSigmoidLambda->getTensorDesc(), &lambda, 1);
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported activation type for Params test type";
    }
}

void ActivationParamLayerTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();
    auto blobInput = inferRequest.GetBlob("Input");
    blobInput = FuncTestUtils::createAndFillBlobFloat(blobInput->getTensorDesc());

    generateActivationBlob();

    inferRequest.Infer();
}


void ActivationParamLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::tie(activationType, netPrecision, shapes, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {shapes.first});
    auto activationParams = createActivationParams(ngPrc);
    params[0]->set_friendly_name("Input");
    params.insert(params.end(), activationParams.begin(), activationParams.end());
    auto activation = ngraph::builder::makeActivation(params, ngPrc, activationType);
    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params);
}

TEST_P(ActivationLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(ActivationParamLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

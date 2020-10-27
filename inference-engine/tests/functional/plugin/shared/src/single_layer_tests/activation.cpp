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
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

std::string ActivationLayerTest::getTestCaseName(const testing::TestParamInfo<activationParams> &obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::string targetDevice;
    std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << activationNames[activationDecl.first] << separator;
    result << "IS=" << CommonTestUtils::vec2str(shapes.first) << separator;
    result << "AS=" << CommonTestUtils::vec2str(shapes.second) << separator;
    result << "ConstantsValue=" << CommonTestUtils::vec2str(activationDecl.second) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "inPRC=" << inPrc.name() << separator;
    result << "outPRC=" << outPrc.name() << separator;
    result << "inL=" << inLayout << separator;
    result << "outL=" << outLayout << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ActivationLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, targetDevice) = GetParam();

    activationType = activationDecl.first;
    auto constantsValue = activationDecl.second;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {shapes.first});
    auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType, shapes.second, constantsValue);

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

void ActivationParamLayerTest::generateActivationBlob(std::vector<float> constantsValue) {
    switch (activationType) {
        case ngraph::helpers::ActivationTypes::PReLu: {
            auto blobNegativeSlope = inferRequest.GetBlob("negativeSlope");
            float negativeSlope = constantsValue[0];
            blobNegativeSlope = FuncTestUtils::createAndFillBlobWithFloatArray(blobNegativeSlope->getTensorDesc(), &negativeSlope, 1);
        }
        case ngraph::helpers::ActivationTypes::LeakyRelu: {
            auto blobLeakySlope = inferRequest.GetBlob("leakySlope");
            float leakySlope = constantsValue[0];
            blobLeakySlope = FuncTestUtils::createAndFillBlobWithFloatArray(blobLeakySlope->getTensorDesc(), &leakySlope, 1);
        }
        case ngraph::helpers::ActivationTypes::HardSigmoid: {
            auto blobHardSigmoidAlpha = inferRequest.GetBlob("alpha");
            auto blobHardSigmoidBeta = inferRequest.GetBlob("beta");
            float alpha = constantsValue[0], beta = constantsValue[1];
            blobHardSigmoidAlpha = FuncTestUtils::createAndFillBlobWithFloatArray(blobHardSigmoidAlpha->getTensorDesc(), &alpha, 1);
            blobHardSigmoidBeta = FuncTestUtils::createAndFillBlobWithFloatArray(blobHardSigmoidBeta->getTensorDesc(), &beta, 1);
        }
        case ngraph::helpers::ActivationTypes::Selu: {
            auto blobHardSigmoidAlpha = inferRequest.GetBlob("alpha");
            auto blobHardSigmoidLambda = inferRequest.GetBlob("lambda");
            float alpha = constantsValue[0], lambda = constantsValue[1];
            blobHardSigmoidAlpha = FuncTestUtils::createAndFillBlobWithFloatArray(blobHardSigmoidAlpha->getTensorDesc(), &alpha, 1);
            blobHardSigmoidLambda = FuncTestUtils::createAndFillBlobWithFloatArray(blobHardSigmoidLambda->getTensorDesc(), &lambda,
  1);
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

    generateActivationBlob(constantsValue);

    inferRequest.Infer();
}


void ActivationParamLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, targetDevice) = GetParam();

    activationType = activationDecl.first;
    constantsValue = activationDecl.second;
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

TEST_P(ActivationLayerTest, TestRefNormalizeIE) {
    std::vector<float> inp1(2 * 3 * 4 * 4);
    std::vector<float> weight(1 * 3 * 1 * 1);
    std::vector<float> out_gpu(2 * 3 * 4 * 4);
    std::vector<float> out_cpu(2 * 3 * 4 * 4);


    std::default_random_engine generator(10);
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    auto gen_data = [&distribution, &generator]() { return distribution(generator); };
    std::generate(inp1.begin(), inp1.end(), gen_data);
    std::generate(weight.begin(), weight.end(), gen_data);


    std::cout << "build number: " << GetInferenceEngineVersion()->buildNumber << '\n';


    auto inp = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 4});


    auto ave_pool = std::make_shared<ngraph::op::v1::AvgPool>(inp, ngraph::Strides{1, 1},
                                     ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, ngraph::Shape{1, 1},
                                     true, ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER);


    float epsilon = 1e-10;
    auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{1});
    auto norm = std::make_shared<ngraph::op::NormalizeL2>(ave_pool, axes, epsilon, ngraph::op::EpsMode::ADD);


    std::vector<size_t> shape{1, 3, 1, 1};
    auto weight_node = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape(shape), weight.data());
    auto mul = std::make_shared<ngraph::op::v1::Multiply>(norm, weight_node, ngraph::op::AutoBroadcastType::NUMPY);



    axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    norm = std::make_shared<ngraph::op::NormalizeL2>(mul, axes, epsilon, ngraph::op::EpsMode::ADD);
    mul = std::make_shared<ngraph::op::v1::Multiply>(norm, weight_node, ngraph::op::AutoBroadcastType::NUMPY);


    auto output = std::make_shared<ngraph::op::Result>(mul);
    auto ngraph_function = std::make_shared<ngraph::Function>(ngraph::ResultVector{output},
                           ngraph::ParameterVector{inp});



    auto cnn = InferenceEngine::CNNNetwork(ngraph_function);


    std::vector<std::string> input_names;
    std::vector<std::string> out_names;
    for (const auto& it : cnn.getInputsInfo()) {
        input_names.push_back(it.first);
    }


    for (const auto& it : cnn.getOutputsInfo()) {
        out_names.push_back(it.first);
    }


    std::vector<size_t> inpSize1 = {2, 3, 4, 4};
    std::vector<size_t> outSize  = {2, 3, 4, 4};


    BlobMap inputBlobs;
    BlobMap outputBlobs;
    TensorDesc tensorDescInp1(Precision::FP32, inpSize1, Layout::NCHW);
    TensorDesc tensorDescOut(Precision::FP32, outSize, Layout::ANY);


    inputBlobs[input_names[0]] = make_shared_blob<float>(tensorDescInp1, inp1.data());
    outputBlobs[out_names[0]]  = make_shared_blob<float>(tensorDescOut, out_cpu.data());


    Core ie;
    ExecutableNetwork executable_network = ie.LoadNetwork(cnn, targetDevice);


    InferRequest infer_request = executable_network.CreateInferRequest();
    infer_request.SetInput(inputBlobs);
    infer_request.SetOutput(outputBlobs);
    infer_request.Infer();


    for (float elem : out_cpu) {
       std::cout << elem << std::endl;
    }
    std::cout  << '\n';
}

}  // namespace LayerTestsDefinitions

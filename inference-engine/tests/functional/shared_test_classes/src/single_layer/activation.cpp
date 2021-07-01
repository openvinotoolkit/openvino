// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/activation.hpp"

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
    params[0]->set_friendly_name("Input");

    if (activationType == ngraph::helpers::ActivationTypes::PReLu && constantsValue.empty()) {
        const auto elemnts_count = ngraph::shape_size(shapes.second);
        constantsValue.resize(elemnts_count);
        std::iota(constantsValue.begin(), constantsValue.end(), -10);
    }

    auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType, shapes.second, constantsValue);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params);
}

InferenceEngine::Blob::Ptr ActivationLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    bool inPrcSigned = function->get_parameters()[0]->get_element_type().is_signed();
    int32_t data_start_from;
    uint32_t data_range;
    int32_t resolution;

    switch (activationType) {
        case ngraph::helpers::ActivationTypes::Log: {
            data_start_from = 1;
            data_range = 20;
            resolution = 32768;
            break;
        }
        case ngraph::helpers::ActivationTypes::Sqrt: {
            data_start_from = 0;
            data_range = 20;
            resolution = 32768;
            break;
        }
        case ngraph::helpers::ActivationTypes::Asin: {
            data_start_from = -1;
            data_range = 2;
            resolution = 32768;
            break;
        }
        case ngraph::helpers::ActivationTypes::Acos: {
            data_start_from = -1;
            data_range = 2;
            resolution = 32768;
            break;
        }
        case ngraph::helpers::ActivationTypes::Ceiling: {
            data_start_from = -1000;
            data_range = 2000;
            resolution = 32768;
            break;
        }
        case ngraph::helpers::ActivationTypes::RoundHalfToEven: {
            data_start_from = -10;
            data_range = 20;
            resolution = 4;
            break;
        }
        case ngraph::helpers::ActivationTypes::RoundHalfAwayFromZero: {
            data_start_from = -10;
            data_range = 20;
            resolution = 4;
            break;
        }
        case ngraph::helpers::ActivationTypes::Mish: {
            data_start_from = -20;
            data_range = 60;
            resolution = 32768;
            break;
        }
        case ngraph::helpers::ActivationTypes::SoftPlus: {
            data_start_from = -100;
            data_range = 200;
            resolution = 32768;
            break;
        }
        default: {
            data_start_from = -10;
            data_range = 20;
            resolution = 32768;
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
                                            resolution);
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
            IE_THROW() << "Unsupported activation type for Params test type";
    }
}

InferenceEngine::Blob::Ptr ActivationParamLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::Blob::Ptr blobPtr;
    const std::string& name = info.name();
    if (name == "negativeSlope") {
        const auto elemnts_count = ngraph::shape_size(function->get_parameters()[1]->get_shape());
        std::vector<float> param_data(elemnts_count);
        std::iota(param_data.begin(), param_data.end(), -10);
        blobPtr = FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), &param_data[0], elemnts_count);
    } else if (name == "leakySlope") {
        const auto elemnts_count = ngraph::shape_size(function->get_parameters()[1]->get_shape());
        std::vector<float> param_data(elemnts_count, constantsValue[0]);
        blobPtr = FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), &param_data[0], elemnts_count);
    } else if (name == "alpha") {
         blobPtr = FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), &constantsValue[0], 1);
    } else if (name == "beta" || name == "lambda") {
        blobPtr = FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), &constantsValue[1], 1);
    } else {
        blobPtr = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 20, -10, 1);
    }
    return blobPtr;
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
    auto activationParams = createActivationParams(ngPrc, shapes.second);

    params[0]->set_friendly_name("Input");
    params.insert(params.end(), activationParams.begin(), activationParams.end());

    auto activation = ngraph::builder::makeActivation(params, ngPrc, activationType);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(activation)};
    function = std::make_shared<ngraph::Function>(results, params);
}

void ActivationDynamicLayerTest::Run() {
    const auto& params = function->get_parameters();
    ngraph::PartialShape output_shape;

    // make each parameter dimension dynamic with range {1 .. prev_dim * 2}
    for (const auto& parameter : params) {
        auto& dynamic_pshape = parameter->get_partial_shape();
        NGRAPH_CHECK(dynamic_pshape.rank().is_static(),
                     "tests are not prepared to work with dynamically ranked inputs");
        for (size_t i = 0; i < dynamic_pshape.rank().get_length(); ++i) {
            if (static_dims.count(i))
                continue;
            dynamic_pshape[i] = {1, dynamic_pshape[i].get_max_length() * 2};
        }
        parameter->set_partial_shape(dynamic_pshape);
        if (parameter->get_friendly_name() == "Input")
            output_shape = dynamic_pshape;
    }
    function->validate_nodes_and_infer_types();

    const auto& results = function->get_results();
    NGRAPH_CHECK(results.size() == 1);
    ASSERT_EQ(results[0]->get_output_partial_shape(0), output_shape);
    // no inference and checks are done here -- just shape check because we miss CNNNetwork functionality
    // to handle dynamic inputs-outputs and test functionality to generate blob of a certain shape
}

}  // namespace LayerTestsDefinitions

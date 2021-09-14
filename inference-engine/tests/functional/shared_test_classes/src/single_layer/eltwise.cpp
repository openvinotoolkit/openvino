// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/single_layer/eltwise.hpp"
#include "functional_test_utils/partial_shape_utils.hpp"
#include <transformations/serialize.hpp>

namespace LayerTestsDefinitions {

std::string EltwiseLayerTest::getTestCaseName(const testing::TestParamInfo<EltwiseTestParams>& obj) {
    std::vector<std::pair<size_t, size_t>> inputShapes;
    std::vector<std::vector<size_t>> targetShapes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    ngraph::helpers::InputLayerType secondaryInputType;
    CommonTestUtils::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseOpType;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, targetShapes, eltwiseOpType, secondaryInputType, opType, netPrecision, inPrc, outPrc, targetName, additional_config) =
        obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "TS=" << CommonTestUtils::vec2str(targetShapes) << "_";
    results << "eltwiseOpType=" << eltwiseOpType << "_";
    results << "secondaryInputType=" << secondaryInputType << "_";
    results << "opType=" << opType << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "inPRC=" << inPrc.name() << "_";
    results << "outPRC=" << outPrc.name() << "_";
    results << "trgDev=" << targetName;
    return results.str();
}

InferenceEngine::Blob::Ptr EltwiseLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    const auto opType = std::get<2>(GetParam());
    switch (opType) {
        case ngraph::helpers::EltwiseTypes::POWER:
        case ngraph::helpers::EltwiseTypes::FLOOR_MOD:
            return info.getPrecision().is_float() ? FuncTestUtils::createAndFillBlob(
                InferenceEngine::TensorDesc(info.getPrecision(), targetStaticShape, const_cast<InferenceEngine::InputInfo&>(info).getLayout()), 2, 2, 128):
                                                    FuncTestUtils::createAndFillBlob(InferenceEngine::TensorDesc(info.getPrecision(), targetStaticShape,
                                        const_cast<InferenceEngine::InputInfo&>(info).getLayout()), 4, 2);
        case ngraph::helpers::EltwiseTypes::DIVIDE:
            return info.getPrecision().is_float() ? FuncTestUtils::createAndFillBlob(
                InferenceEngine::TensorDesc(info.getPrecision(), targetStaticShape, const_cast<InferenceEngine::InputInfo&>(info).getLayout()), 2, 2, 128):
                                                    FuncTestUtils::createAndFillBlob(
                                                        InferenceEngine::TensorDesc(info.getPrecision(), targetStaticShape,
const_cast<InferenceEngine::InputInfo&>(info).getLayout()), 100, 101);
        case ngraph::helpers::EltwiseTypes::ERF:
            return FuncTestUtils::createAndFillBlob(
                InferenceEngine::TensorDesc(info.getPrecision(), targetStaticShape,
                                        const_cast<InferenceEngine::InputInfo&>(info).getLayout()), 6, -3);
        default:
            return FuncTestUtils::createAndFillBlob(
                InferenceEngine::TensorDesc(info.getPrecision(), targetStaticShape,
                                        const_cast<InferenceEngine::InputInfo&>(info).getLayout()));
    }
}

void EltwiseLayerTest::SetUp() {
    std::vector<std::pair<size_t, size_t>> inputShapes;
    std::vector<std::vector<size_t>> targetShapes;
    InferenceEngine::Precision netPrecision;
    CommonTestUtils::OpType opType;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, targetShapes, eltwiseType, secondaryInputType, opType, netPrecision, inPrc, outPrc, targetDevice, additional_config) =
        this->GetParam();
    ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> inputShape2;
    if (targetShapes.size() == 1) {
        inputShape1 = inputShape2 = targetShapes.front();
    } else if (targetShapes.size() == 2) {
        inputShape1 = targetShapes.front();
        inputShape2 = targetShapes.back();
    } else {
        IE_THROW() << "Incorrect number of input shapes";
    }

    targetStaticShape = inputShape1;
    inputDynamicShape = FuncTestUtils::PartialShapeUtils::vec2partialshape(inputShapes, targetStaticShape);

    configuration.insert(additional_config.begin(), additional_config.end());

    switch (opType) {
        case CommonTestUtils::OpType::SCALAR: {
            shape_input_secondary = std::vector<size_t>({1});
            break;
        }
        case CommonTestUtils::OpType::VECTOR:
            shape_input_secondary = inputShape2;
            break;
        default:
            FAIL() << "Unsupported Secondary operation type";
    }

    if (eltwiseType == ngraph::helpers::EltwiseTypes::DIVIDE ||
        eltwiseType == ngraph::helpers::EltwiseTypes::FLOOR_MOD ||
        eltwiseType == ngraph::helpers::EltwiseTypes::MOD) {
        data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape_input_secondary), 10, 2);
    } else if (eltwiseType == ngraph::helpers::EltwiseTypes::POWER && secondaryInputType == ngraph::helpers::InputLayerType::CONSTANT) {
        data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape_input_secondary), 3);
    }

    function = makeEltwise("Eltwise");
    functionRefs = makeEltwise("EltwiseRefs");
}

std::shared_ptr<ngraph::Function> EltwiseLayerTest::makeEltwise(const std::string& name) {
    auto input = ngraph::builder::makeParams(ngPrc, {inputShape1});
    std::shared_ptr<ngraph::Node> secondaryInput;
    if (eltwiseType == ngraph::helpers::EltwiseTypes::DIVIDE ||
        eltwiseType == ngraph::helpers::EltwiseTypes::FLOOR_MOD ||
        eltwiseType == ngraph::helpers::EltwiseTypes::MOD) {
        std::vector<float> data(ngraph::shape_size(shape_input_secondary));
        data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape_input_secondary), 10, 2);
        secondaryInput = ngraph::builder::makeConstant(ngPrc, shape_input_secondary, data);
    } else if (eltwiseType == ngraph::helpers::EltwiseTypes::POWER && secondaryInputType == ngraph::helpers::InputLayerType::CONSTANT) {
        // to avoid floating point overflow on some platforms, let's fill the constant with small numbers.
        secondaryInput = ngraph::builder::makeConstant<float>(ngPrc, shape_input_secondary, data);
    } else {
        secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, shape_input_secondary);
        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            input.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
        }
    }

    auto eltwise = ngraph::builder::makeEltwise(input[0], secondaryInput, eltwiseType);
    return std::make_shared<ngraph::Function>(eltwise, input, name);
}

} // namespace LayerTestsDefinitions

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/scatter_elements_update.hpp"

#include "openvino/op/scatter_elements_update.hpp"
using ov::op::operator<<;

namespace LayerTestsDefinitions {

std::string ScatterElementsUpdateLayerTest::getTestCaseName(const testing::TestParamInfo<scatterElementsUpdateParamsTuple> &obj) {
    axisShapeInShape shapeDescript;
    InferenceEngine::SizeVector indicesValue;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::string targetName;
    std::tie(shapeDescript, indicesValue, inputPrecision, indicesPrecision, targetName) = obj.param;
    std::ostringstream result;
    result << "InputShape=" << ov::test::utils::vec2str(std::get<0>(shapeDescript)) << "_";
    result << "IndicesShape=" << ov::test::utils::vec2str(std::get<1>(shapeDescript)) << "_";
    result << "Axis=" << std::get<2>(shapeDescript) << "_";
    result << "inPrc=" << inputPrecision.name() << "_";
    result << "idxPrc=" << indicesPrecision.name() << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

std::vector<axisShapeInShape> ScatterElementsUpdateLayerTest::combineShapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& inputShapes) {
    std::vector<axisShapeInShape> resVec;
    for (auto& inputShape : inputShapes) {
        for (auto& item : inputShape.second) {
            for (auto& elt : item.second) {
                resVec.push_back(std::make_tuple(inputShape.first, item.first, elt));
            }
        }
    }
    return resVec;
}

void ScatterElementsUpdateLayerTest::SetUp() {
    InferenceEngine::SizeVector inShape;
    InferenceEngine::SizeVector indicesShape;
    int axis;
    axisShapeInShape shapeDescript;
    InferenceEngine::SizeVector indicesValue;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::tie(shapeDescript, indicesValue, inputPrecision, indicesPrecision, targetDevice) = this->GetParam();
    std::tie(inShape, indicesShape, axis) = shapeDescript;
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto idxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indicesPrecision);
    ngraph::ParameterVector paramVector;
    auto inputParams = std::make_shared<ngraph::opset1::Parameter>(inPrc, ngraph::Shape(inShape));
    paramVector.push_back(inputParams);
    auto updateParams = std::make_shared<ngraph::opset1::Parameter>(inPrc, ngraph::Shape(indicesShape));
    paramVector.push_back(updateParams);
    auto s2d = ngraph::builder::makeScatterElementsUpdate(paramVector[0], idxPrc, indicesShape, indicesValue, paramVector[1], axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2d)};
    function = std::make_shared<ngraph::Function>(results, paramVector, "ScatterElementsUpdate");
}

std::string ScatterElementsUpdate12LayerTest::getTestCaseName(const testing::TestParamInfo<scatterElementsUpdate12ParamsTuple> &obj) {
    axisShapeInShape shapeDescript;
    std::vector<int64_t> indicesValue;
    ov::op::v12::ScatterElementsUpdate::Reduction reduceMode;
    bool useInitVal;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::string targetName;
    std::tie(shapeDescript, indicesValue, reduceMode, useInitVal, inputPrecision, indicesPrecision, targetName) = obj.param;
    std::ostringstream result;
    result << "InputShape=" << ov::test::utils::vec2str(std::get<0>(shapeDescript)) << "_";
    result << "IndicesShape=" << ov::test::utils::vec2str(std::get<1>(shapeDescript)) << "_";
    result << "Axis=" << std::get<2>(shapeDescript) << "_";
    result << "ReduceMode=" << reduceMode << "_";
    result << "UseInitVal=" << useInitVal << "_";
    result << "Indices=" << ov::test::utils::vec2str(indicesValue) << "_";
    result << "inPrc=" << inputPrecision.name() << "_";
    result << "idxPrc=" << indicesPrecision.name() << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void ScatterElementsUpdate12LayerTest::SetUp() {
    InferenceEngine::SizeVector inShape;
    InferenceEngine::SizeVector indicesShape;
    int axis;
    ov::op::v12::ScatterElementsUpdate::Reduction reduceMode;
    bool useInitVal;
    axisShapeInShape shapeDescript;
    std::vector<int64_t> indicesValue;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision indicesPrecision;
    std::tie(shapeDescript, indicesValue, reduceMode, useInitVal, inputPrecision, indicesPrecision, targetDevice) = this->GetParam();
    std::tie(inShape, indicesShape, axis) = shapeDescript;
    const auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    const auto idxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indicesPrecision);
    ov::ParameterVector paramVector;
    const auto inputParams = std::make_shared<ov::op::v0::Parameter>(inPrc, ov::Shape(inShape));
    paramVector.push_back(inputParams);
    const auto updateParams = std::make_shared<ov::op::v0::Parameter>(inPrc, ov::Shape(indicesShape));
    paramVector.push_back(updateParams);

    const auto indicesNode = std::make_shared<ov::op::v0::Constant>(idxPrc, indicesShape, indicesValue);
    const auto axisNode = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32, ov::Shape{},
                                                           std::vector<int>{axis});
    const auto seuNode = std::make_shared<ov::op::v12::ScatterElementsUpdate>(paramVector[0], indicesNode,
        paramVector[1], axisNode, reduceMode, useInitVal);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(seuNode)};
    function = std::make_shared<ov::Model>(results, paramVector, "ScatterElementsUpdate");
}
}  // namespace LayerTestsDefinitions

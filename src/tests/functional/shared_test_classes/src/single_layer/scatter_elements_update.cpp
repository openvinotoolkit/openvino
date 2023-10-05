// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/scatter_elements_update.hpp"

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
    auto paramVectorOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramVector));
    auto s2d = ngraph::builder::makeScatterElementsUpdate(paramVectorOuts[0], idxPrc, indicesShape, indicesValue, paramVectorOuts[1], axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2d)};
    function = std::make_shared<ngraph::Function>(results, paramVector, "ScatterElementsUpdate");
}
}  // namespace LayerTestsDefinitions

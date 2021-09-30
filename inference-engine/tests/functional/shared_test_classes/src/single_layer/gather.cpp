// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gather.hpp"
#include "ie_input_info.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

void GatherLayerTestBase::SetUp(const gatherParamsTuple& params) {
    int axis;
    std::vector<int> indices;
    std::vector<size_t> indicesShape;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(indices, indicesShape, axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = params;
    ASSERT_EQ(ngraph::shape_size(indicesShape), indices.size()) << "Indices vector size and provided indices shape doesn't fit each other";
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto functionParams = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(functionParams));
    auto indicesNode = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape(indicesShape), indices);
    auto axisNode = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape({}), {axis});
    auto gather = std::make_shared<ngraph::opset3::Gather>(paramOuts[0], indicesNode, axisNode);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(gather)};
    function = std::make_shared<ngraph::Function>(results, functionParams, "gather");
}

std::string GatherLayerTest::getTestCaseName(const testing::TestParamInfo<gatherParamsTuple> &obj) {
    int axis;
    std::vector<int> indices;
    std::vector<size_t> indicesShape, inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(indices, indicesShape, axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "axis=" << axis << "_";
    result << "indices=" << CommonTestUtils::vec2str(indices) << "_";
    result << "indicesShape=" << CommonTestUtils::vec2str(indicesShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName << "_";
    return result.str();
}

void GatherLayerTest::SetUp() {
    GatherLayerTestBase::SetUp(GetParam());
}

std::string Gather7LayerTest::getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj) {
    std::tuple<int, int> axis_batchIdx;
    std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> inputShapes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(inputShapes, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
    result << "axis=" << std::get<0>(axis_batchIdx) << "_";
    result << "batchIdx=" << std::get<1>(axis_batchIdx) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName << "_";
    return result.str();
}

void Gather7LayerTest::SetUp() {
    std::tuple<int, int> axis_batchIdx;
    std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShapes, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    axis = std::get<0>(axis_batchIdx);
    int batchIdx = std::get<1>(axis_batchIdx);
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    targetStaticShapes.reserve(inputShapes.second.size());
    for (const auto& staticShape : inputShapes.second) {
        targetStaticShapes.push_back({staticShape});
    }
    inputDynamicShapes = { inputShapes.first };
    ov::Shape inputDataShape = targetStaticShapes.front().front(), indicesShape = targetStaticShapes.front().back();

    ov::ParameterVector functionParams {
        ngraph::builder::makeParams(ngPrc, { {"data", inputDataShape} })[0],
        ngraph::builder::makeParams(ov::element::i32, { {"indices", indicesShape} })[0]
    };
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(functionParams));
    auto axisNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({}), { axis });
    auto gather = std::make_shared<ov::op::v7::Gather>(paramOuts[0], paramOuts[1], axisNode, batchIdx);
    ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(gather) };
    function = std::make_shared<ov::Function>(results, functionParams, "Gather7");
    functionRefs = ov::clone_function(*function);
}

void Gather7LayerTest::GenerateInputs() {
    const auto& inputsInfoMap = executableNetwork.GetInputsInfo();
    const auto  dataInfo = inputsInfoMap.find("data");
    if (dataInfo == inputsInfoMap.end())
        THROW_IE_EXCEPTION << "Gather test function does not contain 'data' input info.";
    const auto& dataDims = dataInfo->second->getTensorDesc().getDims();
    for (const auto &input : inputsInfoMap) {
        const auto &info = input.second;
        InferenceEngine::Blob::Ptr blob;
        if (input.first == "indices") {
            blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), dataDims[axis < 0 ? axis + dataDims.size() : axis] - 1, 0);
        } else {
            blob = GenerateInput(*info);
        }
        inputs.push_back(blob);
    }
}

std::string Gather8LayerTest::getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj) {
    std::tuple<int, int> axis_batchIdx;
    std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> inputShapes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(inputShapes, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
    result << "axis=" << std::get<0>(axis_batchIdx) << "_";
    result << "batchIdx=" << std::get<1>(axis_batchIdx) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName << "_";
    return result.str();
}

void Gather8LayerTest::SetUp() {
    std::tuple<int, int> axis_batchIdx;
    std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShapes, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    int axis = std::get<0>(axis_batchIdx);
    int batchIdx = std::get<1>(axis_batchIdx);
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    targetStaticShapes.reserve(inputShapes.second.size());
    for (const auto& staticShape : inputShapes.second) {
        targetStaticShapes.push_back({staticShape});
    }
    inputDynamicShapes = { inputShapes.first };
    ngraph::Shape inputDataShape = targetStaticShapes.front().front(), indicesShape = targetStaticShapes.front().back();

    auto functionParams = ngraph::builder::makeParams(ngPrc, { inputDataShape });
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(functionParams));
    auto indicesNode = ngraph::builder::makeConstant<int>(ngraph::element::i64, indicesShape, {}, true,
                                                          inputDataShape[axis < 0 ? axis + inputDataShape.size() : axis] - 1,
                                                          1 - static_cast<int>(inputDataShape[axis < 0 ? axis + inputDataShape.size() : axis]));
    auto axisNode = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape({}), { axis });
    auto gather = std::make_shared<ngraph::opset8::Gather>(paramOuts[0], indicesNode, axisNode, batchIdx);
    ngraph::ResultVector results{ std::make_shared<ngraph::opset8::Result>(gather) };
    function = std::make_shared<ngraph::Function>(results, functionParams, "Gather8");
    functionRefs = ngraph::clone_function(*function);
}

}  // namespace LayerTestsDefinitions

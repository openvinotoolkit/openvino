// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gather.hpp"

#include "common_test_utils/node_builders/constant.hpp"

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
    ov::ParameterVector functionParams {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto indicesNode = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape(indicesShape), indices);
    auto axisNode = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape({}), {axis});
    auto gather = std::make_shared<ov::op::v1::Gather>(functionParams[0], indicesNode, axisNode);
    ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(gather)};
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
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "axis=" << axis << "_";
    result << "indices=" << ov::test::utils::vec2str(indices) << "_";
    result << "indicesShape=" << ov::test::utils::vec2str(indicesShape) << "_";
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
    std::vector<int> indices;
    std::vector<size_t> indicesShape, inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(inputShape, indicesShape, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "axis=" << std::get<0>(axis_batchIdx) << "_";
    result << "batchIdx=" << std::get<1>(axis_batchIdx) << "_";
    result << "indicesShape=" << ov::test::utils::vec2str(indicesShape) << "_";
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
    std::vector<size_t> indicesShape;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, indicesShape, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    int axis = std::get<0>(axis_batchIdx);
    int batchIdx = std::get<1>(axis_batchIdx);
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector functionParams {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto indicesNode = ov::test::utils::deprecated::make_constant<int>(ngraph::element::i64, indicesShape, {}, true,
                                                          inputShape[axis < 0 ? axis + inputShape.size() : axis] - 1, 0);
    auto axisNode = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape({}), { axis });
    auto gather = std::make_shared<ov::op::v7::Gather>(functionParams[0], indicesNode, axisNode, batchIdx);
    ngraph::ResultVector results{ std::make_shared<ov::op::v0::Result>(gather) };
    function = std::make_shared<ngraph::Function>(results, functionParams, "gather");
}

std::string Gather8LayerTest::getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj) {
    std::tuple<int, int> axis_batchIdx;
    std::vector<int> indices;
    std::vector<size_t> indicesShape, inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(inputShape, indicesShape, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "indicesShape=" << ov::test::utils::vec2str(indicesShape) << "_";
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
    std::vector<size_t> indicesShape;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, indicesShape, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    int axis = std::get<0>(axis_batchIdx);
    int batchIdx = std::get<1>(axis_batchIdx);
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector functionParams {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto indicesNode = ov::test::utils::deprecated::make_constant<int>(ngraph::element::i64, indicesShape, {}, true,
                                                          inputShape[axis < 0 ? axis + inputShape.size() : axis] - 1,
                                                          -static_cast<int>(inputShape[axis < 0 ? axis + inputShape.size() : axis]));
    auto axisNode = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape({}), { axis });
    auto gather = std::make_shared<ov::op::v8::Gather>(functionParams[0], indicesNode, axisNode, batchIdx);
    ngraph::ResultVector results{ std::make_shared<ov::op::v0::Result>(gather) };
    function = std::make_shared<ngraph::Function>(results, functionParams, "gather");
}

std::string Gather8IndiceScalarLayerTest::getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj) {
    std::tuple<int, int> axis_batchIdx;
    std::vector<int> indices;
    std::vector<size_t> indicesShape, inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(inputShape, indicesShape, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "indicesShape=" << ov::test::utils::vec2str(indicesShape) << "_";
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

void Gather8IndiceScalarLayerTest::SetUp() {
    std::tuple<int, int> axis_batchIdx;
    std::vector<size_t> indicesShape;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, indicesShape, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    int axis = std::get<0>(axis_batchIdx);
    int batchIdx = std::get<1>(axis_batchIdx);
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector functionParams {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto indicesNode = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{}, {inputShape[axis] - 1})->output(0);

    auto axisNode = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape({}), { axis });
    auto gather = std::make_shared<ov::op::v8::Gather>(functionParams[0], indicesNode, axisNode, batchIdx);
    ngraph::ResultVector results{ std::make_shared<ov::op::v0::Result>(gather) };
    function = std::make_shared<ngraph::Function>(results, functionParams, "gather");
}

std::string Gather8withIndicesDataLayerTest::getTestCaseName(const testing::TestParamInfo<gather8withIndicesDataParamsTuple>& obj) {
    gather7ParamsTuple basicParams;
    std::vector<int> indicesData;
    std::tie(basicParams, indicesData) = obj.param;

    std::tuple<int, int> axis_batchIdx;
    std::vector<int> indices;
    std::vector<size_t> indicesShape, inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(inputShape, indicesShape, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = basicParams;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "indicesShape=" << ov::test::utils::vec2str(indicesShape) << "_";
    result << "axis=" << std::get<0>(axis_batchIdx) << "_";
    result << "batchIdx=" << std::get<1>(axis_batchIdx) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName << "_";

    result << "indicesData=" << ov::test::utils::vec2str(indicesData) << "_";

    return result.str();
}

void Gather8withIndicesDataLayerTest::SetUp() {
    gather7ParamsTuple basicParams;
    std::vector<int> indicesData;
    std::tie(basicParams, indicesData) = GetParam();

    std::tuple<int, int> axis_batchIdx;
    std::vector<size_t> indicesShape;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, indicesShape, axis_batchIdx, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = basicParams;
    int axis = std::get<0>(axis_batchIdx);
    int batchIdx = std::get<1>(axis_batchIdx);
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector functionParams {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto indicesNode = ov::test::utils::deprecated::make_constant<int>(ngraph::element::i64, indicesShape, indicesData);
    auto axisNode = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape({}), { axis });
    auto gather = std::make_shared<ov::op::v8::Gather>(functionParams[0], indicesNode, axisNode, batchIdx);
    ngraph::ResultVector results{ std::make_shared<ov::op::v0::Result>(gather) };
    function = std::make_shared<ngraph::Function>(results, functionParams, "gather");
}

}  // namespace LayerTestsDefinitions

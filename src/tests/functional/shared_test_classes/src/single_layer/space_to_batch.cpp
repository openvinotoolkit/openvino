// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/space_to_batch.hpp"

namespace LayerTestsDefinitions {

std::string SpaceToBatchLayerTest::getTestCaseName(const testing::TestParamInfo<spaceToBatchParamsTuple> &obj) {
    std::vector<size_t> inShapes;
    std::vector<int64_t> blockShape, padsBegin, padsEnd;
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(blockShape, padsBegin, padsEnd, inShapes, netPrc, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inShapes) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "BS=" << ov::test::utils::vec2str(blockShape) << "_";
    result << "PB=" << ov::test::utils::vec2str(padsBegin) << "_";
    result << "PE=" << ov::test::utils::vec2str(padsEnd) << "_";
    result << "trgDev=" << targetName << "_";
    return result.str();
}

void SpaceToBatchLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<int64_t> blockShape, padsBegin, padsEnd;
    InferenceEngine::Precision netPrecision;
    std::tie(blockShape, padsBegin, padsEnd, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto s2b = ngraph::builder::makeSpaceToBatch(params[0], ngPrc, blockShape, padsBegin, padsEnd);
    OPENVINO_SUPPRESS_DEPRECATED_END
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2b)};
    function = std::make_shared<ngraph::Function>(results, params, "SpaceToBatch");
}
}  // namespace LayerTestsDefinitions

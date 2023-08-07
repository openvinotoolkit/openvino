// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/reverse_sequence.hpp"

namespace LayerTestsDefinitions {
std::string ReverseSequenceLayerTest::getTestCaseName(const testing::TestParamInfo<ReverseSequenceParamsTuple> &obj) {
    int64_t batchAxisIndx;
    int64_t seqAxisIndx;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::vector<size_t> inputShape;
    std::vector<size_t> secondInputShape;
    ngraph::helpers::InputLayerType secondaryInputType;

    std::tie(batchAxisIndx, seqAxisIndx, inputShape, secondInputShape, secondaryInputType, netPrecision, targetName) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "seqLengthsShape" << ov::test::utils::vec2str(secondInputShape) << "_";
    result << "secondaryInputType=" << secondaryInputType << "_";
    result << "batchAxis=" << batchAxisIndx << "_";
    result << "seqAxis=" << seqAxisIndx << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void ReverseSequenceLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    int64_t batchAxisIndx;
    int64_t seqAxisIndx;
    std::vector<size_t> inputShape;
    std::vector<size_t> secondInputShape;
    ngraph::helpers::InputLayerType secondaryInputType;

    std::tie(batchAxisIndx, seqAxisIndx, inputShape, secondInputShape, secondaryInputType, netPrecision, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ov::test::utils::builder::makeParams(ngPrc, {inputShape});

    auto secondPrc = ov::element::i32; //according to the specification
    std::shared_ptr<ngraph::Node> secondaryInput;
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        secondaryInput = std::make_shared<ov::op::v0::Parameter>(secondPrc, ov::Shape(secondInputShape));
        paramsIn.push_back(std::static_pointer_cast<ov::op::v0::Parameter>(secondaryInput));
    } else {
        secondaryInput = std::make_shared<ov::op::v0::Constant>(secondPrc, ov::Shape(secondInputShape));
    }

    auto reverse = std::make_shared<ngraph::opset1::ReverseSequence>(paramsIn[0], secondaryInput, batchAxisIndx, seqAxisIndx);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reverse)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "ReverseSequence");
}

} // namespace LayerTestsDefinitions

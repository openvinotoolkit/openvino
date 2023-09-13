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
    ov::ParameterVector paramsIn {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto secondPrc = ngraph::element::Type_t::i32; //according to the specification
    auto secondaryInput = ngraph::builder::makeInputLayer(secondPrc, secondaryInputType, secondInputShape);
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        paramsIn.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
    }

    auto reverse = std::make_shared<ngraph::opset1::ReverseSequence>(paramsIn[0], secondaryInput, batchAxisIndx, seqAxisIndx);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reverse)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "ReverseSequence");
}

} // namespace LayerTestsDefinitions

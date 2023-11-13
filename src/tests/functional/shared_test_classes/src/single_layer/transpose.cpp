// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/transpose.hpp"

namespace LayerTestsDefinitions {

std::string TransposeLayerTest::getTestCaseName(const testing::TestParamInfo<transposeParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::vector<size_t> inputShapes, inputOrder;
    std::string targetDevice;
    std::tie(inputOrder, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "inputOrder=" << ov::test::utils::vec2str(inputOrder) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void TransposeLayerTest::SetUp() {
    std::vector<size_t> inputShape, inputOrder;
    InferenceEngine::Precision netPrecision;
    std::tie(inputOrder, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto inOrderShape = inputOrder.empty() ? ngraph::Shape({0}) : ngraph::Shape({inputShape.size()});
    const auto inputOrderOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64,
                                                                         inOrderShape,
                                                                         inputOrder);
    const auto transpose = std::make_shared<ngraph::opset3::Transpose>(paramOuts.at(0), inputOrderOp);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(transpose)};
    function = std::make_shared<ngraph::Function>(results, params, "Transpose");
}

}  // namespace LayerTestsDefinitions

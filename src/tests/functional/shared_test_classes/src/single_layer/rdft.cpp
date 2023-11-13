// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/rdft.hpp"

namespace LayerTestsDefinitions {

std::string RDFTLayerTest::getTestCaseName(const testing::TestParamInfo<RDFTParams>& obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::vector<int64_t> axes;
    std::vector<int64_t> signalSize;
    ngraph::helpers::DFTOpType opType;
    std::string targetDevice;
    std::tie(inputShapes, inputPrecision, axes, signalSize, opType, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "Precision=" << inputPrecision.name() << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "SignalSize=" << ov::test::utils::vec2str(signalSize) << "_";
    result << "Inverse=" << (opType == ngraph::helpers::DFTOpType::INVERSE) << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void RDFTLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::vector<int64_t> axes;
    std::vector<int64_t> signalSize;
    ngraph::helpers::DFTOpType opType;
    std::tie(inputShapes, inputPrecision, axes, signalSize, opType, targetDevice) = this->GetParam();
    auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    ngraph::ParameterVector paramVector;
    auto paramData = std::make_shared<ngraph::opset1::Parameter>(inType, ngraph::Shape(inputShapes));
    paramVector.push_back(paramData);

    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramVector));
    auto rdft = ngraph::builder::makeRDFT(paramOuts[0], axes, signalSize, opType);


    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(rdft)};
    function = std::make_shared<ngraph::Function>(results, paramVector, "RDFT");
}
}  // namespace LayerTestsDefinitions

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/dft.hpp"

namespace LayerTestsDefinitions {

std::string DFTLayerTest::getTestCaseName(const testing::TestParamInfo<DFTParams>& obj) {
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

void DFTLayerTest::SetUp() {
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

    auto dft = ngraph::builder::makeDFT(paramVector[0], axes, signalSize, opType);


    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(dft)};
    function = std::make_shared<ngraph::Function>(results, paramVector, "DFT");
}
}  // namespace LayerTestsDefinitions

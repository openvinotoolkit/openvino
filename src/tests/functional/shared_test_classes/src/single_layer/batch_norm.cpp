// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/batch_norm.hpp"

namespace LayerTestsDefinitions {
std::string BatchNormLayerTest::getTestCaseName(const testing::TestParamInfo<BatchNormLayerTestParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    double epsilon;
    std::string targetDevice;
    std::tie(epsilon, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "epsilon=" << epsilon << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr BatchNormLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), 3, 0, 1);
}

void BatchNormLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    double epsilon;
    std::tie(epsilon, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));

    auto batchNorm = ngraph::builder::makeBatchNormInference(paramOuts[0], epsilon);
    ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(batchNorm)};
    function = std::make_shared<ngraph::Function>(results, params, "BatchNormInference");
}

}  // namespace LayerTestsDefinitions

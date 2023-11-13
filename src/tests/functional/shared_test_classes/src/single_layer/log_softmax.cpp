// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/log_softmax.hpp"

namespace LayerTestsDefinitions {

std::string LogSoftmaxLayerTest::getTestCaseName(const testing::TestParamInfo<logSoftmaxLayerTestParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShape;
    int64_t axis;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, axis, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "axis=" << axis << "_";
    result << "trgDev=" << targetDevice;

    return result.str();
}

void LogSoftmaxLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    int64_t axis;

    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, axis, targetDevice, configuration) = GetParam();
    outLayout = inLayout;

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    const auto logSoftmax = std::make_shared<ngraph::op::v5::LogSoftmax>(params.at(0), axis);

    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(logSoftmax)};

    function = std::make_shared<ngraph::Function>(results, params, "logSoftmax");
}
}  // namespace LayerTestsDefinitions

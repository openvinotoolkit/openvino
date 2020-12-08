// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/log_softmax.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "ie_core.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

namespace LayerTestsDefinitions {

std::string LogSoftmaxLayerTest::getTestCaseName(testing::TestParamInfo<logSoftmaxLayerTestParams> obj) {
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
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
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

    const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    const auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto logSoftmax = std::make_shared<ngraph::op::v5::LogSoftmax>(paramOuts.at(0), axis);

    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(logSoftmax)};

    function = std::make_shared<ngraph::Function>(results, params, "logSoftmax");
}

TEST_P(LogSoftmaxLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

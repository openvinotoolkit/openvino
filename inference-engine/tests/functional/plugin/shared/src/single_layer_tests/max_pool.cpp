// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/max_pool.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "ie_core.hpp"

#include "ngraph/op/max_pool.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

namespace LayerTestsDefinitions {

std::string MaxPoolLayerTest::getTestCaseName(testing::TestParamInfo<MaxPoolLayerTestParams> obj) {
    MaxPoolSpecificParams params;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(params, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "inShape=" << CommonTestUtils::vec2str(params.inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "pads_begin=" << CommonTestUtils::vec2str(params.pads_begin) << "_";
    result << "pads_end=" << CommonTestUtils::vec2str(params.pads_end) << "_";
    result << "strides=" << CommonTestUtils::vec2str(params.strides) << "_";
    result << "kernel=" << CommonTestUtils::vec2str(params.kernel) << "_";
    result << "rounding_type=" << params.rounding_type << "_";
    result << "pad_type=" << params.pad_type << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void MaxPoolLayerTest::SetUp() {
    MaxPoolSpecificParams ssParams;
    InferenceEngine::Precision netPrecision;
    std::tie(ssParams, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, configuration) = GetParam();
    outLayout = inLayout;
    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    const auto params = ngraph::builder::makeParams(ngPrc, {ssParams.inputShape});
    const auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    const auto MaxPool = std::make_shared<ngraph::opset1::MaxPool>(paramOuts.at(0), ssParams.strides, ssParams.pads_begin,
    ssParams.pads_end, ssParams.kernel, ssParams.rounding_type, ssParams.pad_type);
    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(MaxPool)};
    function = std::make_shared<ngraph::Function>(results, params, "MaxPool");
}

TEST_P(MaxPoolLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

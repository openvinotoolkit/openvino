// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/softmax.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "ie_core.hpp"

#include "ngraph/op/softmax.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

namespace LayerTestsDefinitions {

std::string SoftMaxLayerTest::getTestCaseName(testing::TestParamInfo<softMaxLayerTestParams> obj) {
    InferenceEngine::Precision netPrecision, inputPrecision;
    InferenceEngine::Layout inputLayout;
    InferenceEngine::SizeVector inputShape;
    size_t axis;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inputPrecision, inputLayout, inputShape, axis, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "inLayout=" << inputLayout << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "axis=" << axis << "_";
    result << "targetDevice=" << targetDevice;

    return result.str();
}

void SoftMaxLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    size_t axis;
    std::tie(netPrecision, inputPrecision, inputLayout, inputShape, axis, targetDevice, config) = GetParam();
    outputPrecision = inputPrecision;
    outputLayout = inputLayout;

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    const auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);

    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(softMax)};

    fnPtr = std::make_shared<ngraph::Function>(results, params, "softMax");
}

TEST_P(SoftMaxLayerTest, CompareWithRefs) {
    inferAndValidate();
}

}  // namespace LayerTestsDefinitions

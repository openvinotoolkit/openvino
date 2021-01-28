// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/one_hot.hpp"

namespace LayerTestsDefinitions {

std::string OneHotLayerTest::getTestCaseName(testing::TestParamInfo<oneHotLayerTestParamsSet> obj) {
    int64_t depth, axis;
    float on_value, off_value;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::SizeVector inputShape;
    LayerTestsUtils::TargetDevice targetDevice;

    std::tie(depth, on_value, off_value, axis, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
      obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "depth=" << depth << "_";
    result << "onValue=" << on_value << "_";
    result << "offValue=" << off_value << "_";
    result << "axis=" << axis << "_";

    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void OneHotLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    int64_t depth, axis;
    float on_value, off_value;
    InferenceEngine::Precision netPrecision;
    std::tie(depth, on_value, off_value, axis, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
    this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));

    auto onehot = ngraph::builder::makeOneHot(paramOuts[0], depth, on_value, off_value, axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(onehot)};
    function = std::make_shared<ngraph::Function>(results, params, "OneHot");
}
}  // namespace LayerTestsDefinitions
// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/one_hot.hpp"

namespace LayerTestsDefinitions {

std::string OneHotLayerTest::getTestCaseName(testing::TestParamInfo<oneHotLayerTestParamsSet> obj) {
    int64_t axis;
    std::pair<ngraph::element::Type, int64_t> depth_type_val;
    std::pair<ngraph::element::Type, float> on_type_val, off_type_val;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::SizeVector inputShape;
    LayerTestsUtils::TargetDevice targetDevice;

    std::tie(depth_type_val, on_type_val, off_type_val, axis, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
      obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "depthType=" << depth_type_val.first << "_";
    result << "depth=" << depth_type_val.second << "_";
    result << "onValueType=" << on_type_val.first << "_";
    result << "onValue=" << on_type_val.second << "_";
    result << "offValueType=" << off_type_val.first << "_";
    result << "offValue=" << off_type_val.second << "_";
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
    int64_t axis;
    std::pair<ngraph::element::Type, int64_t> depth_type_val;
    std::pair<ngraph::element::Type, float> on_type_val, off_type_val;
    InferenceEngine::Precision netPrecision;
    std::tie(depth_type_val, on_type_val, off_type_val, axis, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
    this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));

    auto onehot = ngraph::builder::makeOneHot(paramOuts[0], depth_type_val, on_type_val, off_type_val, axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(onehot)};
    function = std::make_shared<ngraph::Function>(results, params, "OneHot");
}
}  // namespace LayerTestsDefinitions
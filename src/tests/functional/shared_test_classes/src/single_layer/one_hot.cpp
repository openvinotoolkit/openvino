// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/one_hot.hpp"

namespace LayerTestsDefinitions {

std::string OneHotLayerTest::getTestCaseName(const testing::TestParamInfo<oneHotLayerTestParamsSet>& obj) {
    int64_t axis;
    ngraph::element::Type depth_type, set_type;
    int64_t depth_val;
    float on_val, off_val;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    LayerTestsUtils::TargetDevice targetDevice;

    std::tie(depth_type, depth_val, set_type, on_val, off_val, axis, netPrecision, inputShape, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "depthType=" << depth_type << "_";
    result << "depth=" << depth_val << "_";
    result << "SetValueType=" << set_type << "_";
    result << "onValue=" << on_val << "_";
    result << "offValue=" << off_val << "_";
    result << "axis=" << axis << "_";

    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void OneHotLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    int64_t axis;
    ngraph::element::Type depth_type, set_type;
    int64_t depth_val;
    float on_val, off_val;
    InferenceEngine::Precision netPrecision;
    std::tie(depth_type, depth_val, set_type, on_val, off_val, axis, netPrecision, inputShape, targetDevice) =
    this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));

    auto onehot = ngraph::builder::makeOneHot(paramOuts[0], depth_type, depth_val, set_type, on_val, off_val, axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(onehot)};
    function = std::make_shared<ngraph::Function>(results, params, "OneHot");
}
}  // namespace LayerTestsDefinitions

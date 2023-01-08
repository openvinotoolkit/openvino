// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/broadcast.hpp"

namespace LayerTestsDefinitions {
std::string BroadcastLayerTest::getTestCaseName(const testing::TestParamInfo<BroadcastParamsTuple>& obj) {
    InferenceEngine::SizeVector targetShape;
    ov::AxisSet axesMapping;
    ov::op::BroadcastType mode;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision networkPrecision;
    std::string deviceName;
    std::tie(targetShape, axesMapping, mode, inputShape, networkPrecision, deviceName) = obj.param;

    std::ostringstream result;
    result << "targetShape=" << CommonTestUtils::vec2str(targetShape) << "_";
    result << "axesMapping=" << CommonTestUtils::set2str(axesMapping)  << "_";
    result << "mode=" << mode << "_";
    result << "inShape=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "inNPrec=" << networkPrecision << "_";
    result << "trgDev=" << deviceName;
    return result.str();
}

void BroadcastLayerTest::SetUp() {
    InferenceEngine::SizeVector targetShape;
    ov::AxisSet axesMapping;
    ov::op::BroadcastType mode;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision networkPrecision;
    std::tie(targetShape, axesMapping, mode, inputShape, networkPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(networkPrecision);

    auto target_shape_const = ov::op::v0::Constant::create(ov::element::i64, {targetShape.size()}, targetShape);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto broadcast = ngraph::builder::makeBroadcast(params[0], target_shape_const, mode, axesMapping);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(broadcast)};
    function = std::make_shared<ov::Model>(results, params, "BroadcastInference");
}

}  // namespace LayerTestsDefinitions

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/shuffle_channels.hpp"

namespace LayerTestsDefinitions {

std::string ShuffleChannelsLayerTest::getTestCaseName(const testing::TestParamInfo<shuffleChannelsLayerTestParamsSet>& obj) {
    shuffleChannelsSpecificParams shuffleChannelsParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(shuffleChannelsParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
    int axis, group;
    std::tie(axis, group) = shuffleChannelsParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "Axis=" << std::to_string(axis) << "_";
    result << "Group=" << std::to_string(group) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ShuffleChannelsLayerTest::SetUp() {
    shuffleChannelsSpecificParams shuffleChannelsParams;
    std::vector<size_t> inputShape;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(shuffleChannelsParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
    int axis, group;
    std::tie(axis, group) = shuffleChannelsParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto shuffleChannels = std::dynamic_pointer_cast<ngraph::opset3::ShuffleChannels>(
            ngraph::builder::makeShuffleChannels(params[0], axis, group));
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(shuffleChannels)};
    function = std::make_shared<ngraph::Function>(results, params, "shuffleChannels");
}
}  // namespace LayerTestsDefinitions

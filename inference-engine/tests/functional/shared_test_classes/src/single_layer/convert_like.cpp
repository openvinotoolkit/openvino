// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convert_like.hpp"

namespace LayerTestsDefinitions {

std::string ConvertLikeLayerTest::getTestCaseName(const testing::TestParamInfo<ConvertLikeParamsTuple> &obj) {
    InferenceEngine::Precision precision, targetPrecision;
    InferenceEngine::Layout inLayout, outLayout;
    std::vector<std::vector<size_t>> inputShape1, inputShape2;
    std::string targetName;
    std::tie(inputShape1, precision, inputShape2, targetPrecision, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS1=" << CommonTestUtils::vec2str(inputShape1) << "_";
    result << "IS2=" << CommonTestUtils::vec2str(inputShape2) << "_";
    result << "PRC1=" << precision.name() << "_";
    result << "PRC2=" << targetPrecision.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ConvertLikeLayerTest::SetUp() {
    InferenceEngine::Precision inputPrecision, targetPrecision;
    std::vector<std::vector<size_t>> inputShape1, inputShape2;
    std::tie(inputShape1, inputPrecision, inputShape2, targetPrecision, inLayout, outLayout, targetDevice) = GetParam();
    auto ngPrc1 = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto targetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(targetPrecision);
    auto params = ngraph::builder::makeParams(ngPrc1, inputShape1);
    params.push_back(ngraph::builder::makeParams(targetPrc, inputShape2).front());
    auto convertLike = std::make_shared<ngraph::opset3::ConvertLike>(params.front(), params.back());
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(convertLike)};
    function = std::make_shared<ngraph::Function>(results, params, "ConvertLike");
}
}  // namespace LayerTestsDefinitions
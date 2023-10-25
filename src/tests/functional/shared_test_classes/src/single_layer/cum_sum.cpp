// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/cum_sum.hpp"

namespace LayerTestsDefinitions {

std::string CumSumLayerTest::getTestCaseName(const testing::TestParamInfo<cumSumParams>& obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    int64_t axis;
    bool exclusive, reverse;
    std::string targetDevice;
    std::tie(inputShapes, inputPrecision, axis, exclusive, reverse, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "Precision=" << inputPrecision.name() << "_";
    result << "Axis=" << axis << "_";
    result << "Exclusive=" << (exclusive ? "TRUE" : "FALSE") << "_";
    result << "Reverse=" << (reverse ? "TRUE" : "FALSE") << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void CumSumLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    bool exclusive, reverse;
    int64_t axis;
    std::tie(inputShapes, inputPrecision, axis, exclusive, reverse, targetDevice) = this->GetParam();
    const auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    const auto paramData = std::make_shared<ngraph::op::Parameter>(inType, ngraph::Shape(inputShapes));
    const auto axisNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<int64_t>{axis})->output(0);
    const auto cumSum = std::make_shared<ngraph::op::v0::CumSum>(paramData, axisNode, exclusive, reverse);

    ngraph::ResultVector results{std::make_shared<ngraph::op::Result>(cumSum)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{paramData}, "cumsum");
}
}  // namespace LayerTestsDefinitions

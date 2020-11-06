// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/range_add.hpp"

namespace LayerTestsDefinitions {

std::string RangeAddSubgraphTest::getTestCaseName(testing::TestParamInfo<RangeParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    float start, stop, step;
    std::string targetDevice;
    std::tie(start, stop, step, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "Start=" << start << separator;
    result << "Stop=" << stop << separator;
    result << "Step=" << step << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void RangeAddSubgraphTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    float start, stop, step;
    std::tie(start, stop, step, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto startConstant = std::make_shared<ngraph::opset1::Constant>(ngPrc, ngraph::Shape{}, start);
    auto stopConstant = std::make_shared<ngraph::opset1::Constant>(ngPrc, ngraph::Shape{}, stop);
    auto stepConstant = std::make_shared<ngraph::opset1::Constant>(ngPrc, ngraph::Shape{}, step);
    auto range = std::make_shared<ngraph::opset3::Range>(startConstant, stopConstant, stepConstant);

    auto params = ngraph::builder::makeParams(ngPrc, {range->get_shape()});
    auto eltwise = ngraph::builder::makeEltwise(params.front(), range, ngraph::helpers::EltwiseTypes::ADD);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(eltwise)};
    function = std::make_shared<ngraph::Function>(results, params, "RangeEltwise");
}

TEST_P(RangeAddSubgraphTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
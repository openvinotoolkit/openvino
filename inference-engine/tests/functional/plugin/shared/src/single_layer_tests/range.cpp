// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/range.hpp"

namespace LayerTestsDefinitions {
std::string RangeLayerTest::getTestCaseName(testing::TestParamInfo<RangeParams> obj) {
    InferenceEngine::Precision netPrecision;
    float start, stop, step;
    std::string targetDevice;
    std::tie(start, stop, step, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "Start=" << start << separator;
    result << "Stop=" << stop << separator;
    result << "Step=" << step << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void RangeLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    float start, stop, step;
    std::tie(start, stop, step, netPrecision, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    std::vector<size_t> inShape;
    auto params = ngraph::builder::makeParams(ngPrc, {inShape});
    auto start_constant = std::make_shared<ngraph::opset1::Constant>(ngPrc, ngraph::Shape{}, start);
    auto stop_constant = std::make_shared<ngraph::opset1::Constant>(ngPrc, ngraph::Shape{}, stop);
    auto step_constant = std::make_shared<ngraph::opset1::Constant>(ngPrc, ngraph::Shape{}, step);
    auto range = std::make_shared<ngraph::opset3::Range>(start_constant, stop_constant, step_constant);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(range)};
    function = std::make_shared<ngraph::Function>(results, params, "Range");
}

TEST_P(RangeLayerTest, CompareWithRefs) {
    Run();
}

} // namespace LayerTestsDefinitions
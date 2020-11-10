// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/broadcast_power.hpp"

namespace LayerTestsDefinitions {
std::string BroadcastPowerTest::getTestCaseName(testing::TestParamInfo<BroadCastPowerTuple> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::vector<std::vector<size_t>> inputs_shapes;
    std::tie(inputs_shapes, netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "inputShape=" << CommonTestUtils::vec2str(inputs_shapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void BroadcastPowerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::vector<std::vector<size_t>> inputs_shapes;
    std::tie(inputs_shapes, netPrecision, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputs_shapes[0]});
    auto reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{inputs_shapes[1].size()},
            inputs_shapes[1]);
    auto reshape = std::make_shared<ngraph::opset1::Reshape>(params[0], reshape_pattern, false);

    auto const_mult2 = ngraph::builder::makeConstant<float>(ngPrc, {}, {-1.0f});
    auto sum = ngraph::builder::makeEltwise(reshape, const_mult2, ngraph::helpers::EltwiseTypes::MULTIPLY);

    auto reshape_pattern_2 = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{inputs_shapes[0].size()},
            inputs_shapes[0]);
    auto reshape_2 = std::make_shared<ngraph::opset1::Reshape>(sum, reshape_pattern_2, false);
    function = std::make_shared<ngraph::Function>(reshape_2, params, "BroadcastPowerPass");
}

TEST_P(BroadcastPowerTest, CompareWithRefImpl) {
    Run();
};
} // namespace LayerTestsDefinitions

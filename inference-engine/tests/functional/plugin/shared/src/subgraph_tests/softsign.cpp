// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <ie_plugin_config.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include <legacy/ngraph_ops/power.hpp>
#include "subgraph_tests/softsign.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string SoftsignTest::getTestCaseName(testing::TestParamInfo<softsignParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShape;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void SoftsignTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });

    auto abs = std::make_shared<ngraph::op::Abs>(params[0]);
    auto add = std::make_shared<ngraph::op::PowerIE>(abs, 1, 1, 1);
    auto power = std::make_shared<ngraph::op::PowerIE>(add, -1, 1, 0);
    auto mul = std::make_shared<ngraph::op::Multiply>(power, params[0]);
    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(mul) };
    function = std::make_shared<ngraph::Function>(results, params, "SoftSignTest");
}

void SoftsignTest::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LoadNetwork();
    Infer();

    function = GenerateNgraphFriendlySoftSign();
    Validate();
}

std::shared_ptr<ngraph::Function> SoftsignTest::GenerateNgraphFriendlySoftSign() {
    InferenceEngine::Precision netPrecision = std::get<0>(this->GetParam());
    std::vector<size_t> inputShape = std::get<3>(this->GetParam());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });
    auto abs = std::make_shared<ngraph::op::Abs>(params[0]);
    auto constant_0 = ngraph::builder::makeConstant<float>(ngPrc, inputShape, { 1 });
    auto add = std::make_shared<ngraph::op::Add>(abs, constant_0);
    auto constant_1 = ngraph::builder::makeConstant<float>(ngPrc, inputShape, { -1 });
    auto power = std::make_shared<ngraph::op::Power>(add, constant_1);
    auto mul = std::make_shared<ngraph::op::Multiply>(power, params[0]);

    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(mul) };
    return std::make_shared<ngraph::Function>(results, params, "SoftSignTest");
}

TEST_P(SoftsignTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions

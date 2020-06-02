// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"

#include "subgraph_tests/concat_quantization.hpp"


namespace LayerTestsDefinitions {

std::string ConcatQuantization::getTestCaseName(testing::TestParamInfo<concatQuantizationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }

    return result.str();
}

void ConcatQuantization::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {{1, 160}});

    std::vector<size_t> outFormShapes1 = { 1, 5, 32 };
    auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 3 }, outFormShapes1);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

    auto tanh = std::make_shared<ngraph::opset1::Tanh>(reshape1);

    std::vector<size_t> outFormShapes2 = { 1, 160 };
    auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes2);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(tanh, pattern2, false);
    auto scale = ngraph::builder::makeConstant(ngPrc, outFormShapes2, {}, true);
    //For ngraph::op::ScaleShift: Cannot cast ngraph node ScaleShift to CNNLayer!
    auto scale_shift = std::make_shared<ngraph::opset1::Multiply>(reshape2, scale);

    std::vector<size_t> outFormShapes3 = { 5, 32 };
    auto pattern3 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes3);
    auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(scale_shift, pattern3, false);

    auto pattern4 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes3);
    auto reshape4 = std::make_shared<ngraph::opset1::Reshape>(tanh, pattern4, false);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ reshape3, reshape4 }, 0);
    concat->set_friendly_name("concat");

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "ConcatQuantization");
}

TEST_P(ConcatQuantization, CompareWithRefImpl) {
    InferenceEngine::Core* core = PluginCache::get().ie(targetDevice).get();
    if (!configuration.empty()) {
        core->SetConfig(configuration, targetDevice);
    }

    try {
        InferenceEngine::CNNNetwork cnnNetwork = InferenceEngine::CNNNetwork{ function };
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice);
    }
    catch (InferenceEngine::details::InferenceEngineException ex) {
        FAIL() << ex.what();
    }

    if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
};

}  // namespace LayerTestsDefinitions

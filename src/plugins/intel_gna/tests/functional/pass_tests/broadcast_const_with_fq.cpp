// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#include <ie_core.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using BroadcastConstWithFqParamsTuple = typename std::tuple<InferenceEngine::Precision,  // Network Precision
                                                            std::vector<size_t>,  // Input shapes for Params Layer
                                                            std::vector<size_t>,  // Input shapes for Constant Layer
                                                            size_t,               // Quantization level
                                                            std::map<std::string, std::string>,  // Configuration
                                                            std::string>;                        // Device name

namespace LayerTestsDefinitions {

class BroadcastConstWithFq : public testing::WithParamInterface<BroadcastConstWithFqParamsTuple>,
                             public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BroadcastConstWithFqParamsTuple> obj) {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape1;
        std::vector<size_t> inputShape2;
        size_t level{0};
        std::map<std::string, std::string> configuration;
        std::string targetDevice;
        std::tie(netPrecision, inputShape1, inputShape2, level, configuration, targetDevice) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "configItem=" << configItem.first << "_" << configItem.second << "_";
        }
        result << "inputShape1=" << ov::test::utils::vec2str(inputShape1) << "_";
        result << "inputShape2=" << ov::test::utils::vec2str(inputShape2) << "_";
        result << "level=" << level;
        return result.str();
    }

protected:
    void SetUp() override {
        size_t level{0};
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape1;
        std::vector<size_t> inputShape2;
        std::tie(netPrecision, inputShape1, inputShape2, level, configuration, targetDevice) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape1))};
        auto fakeQuantize1 =
            ngraph::builder::makeFakeQuantize(params[0], ngPrc, level, {}, {-0.5}, {0.5}, {-0.5}, {0.5});
        auto constant = ngraph::builder::makeConstant<float>(ngPrc, inputShape2, {}, true);
        auto fakeQuantize2 =
            ngraph::builder::makeFakeQuantize(constant, ngPrc, level, {}, {-0.5}, {0.5}, {-0.5}, {0.5});
        auto add = std::make_shared<ngraph::opset1::Add>(fakeQuantize1, fakeQuantize2);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
        function = std::make_shared<ngraph::Function>(results, params, "BroadcastConstWithFq");
    }
};

TEST_P(BroadcastConstWithFq, CompareWithRefImpl) {
    Run();
};

std::vector<std::vector<size_t>> inputShapes1 = {{1, 1, 21, 160}};
std::vector<std::vector<size_t>> inputShapes2 = {{1, 1, 1, 160}};
const std::vector<size_t> level = {65535};
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};
const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

INSTANTIATE_TEST_SUITE_P(smoke_broadcast_const_with_fq,
                         BroadcastConstWithFq,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(inputShapes1),
                                            ::testing::ValuesIn(inputShapes2),
                                            ::testing::ValuesIn(level),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         BroadcastConstWithFq::getTestCaseName);
}  // namespace LayerTestsDefinitions

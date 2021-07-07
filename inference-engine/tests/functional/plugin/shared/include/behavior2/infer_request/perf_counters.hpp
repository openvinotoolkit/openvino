// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
class PerfCountersTest : public BehaviorTestsUtils::InferRequestTests {
    void SetUp() override {
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
        configuration.insert({ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES });
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    }
};

TEST_P(PerfCountersTest, NotEmptyWhenExecuted) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::CNNNetwork cnnNet(function);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::ResponseDesc response;
    ASSERT_NO_FATAL_FAILURE(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(perfMap = req.GetPerformanceCounts());
    ASSERT_NE(perfMap.size(), 0);
}
}  // namespace BehaviorTestsDefinitions
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ie_preprocess.hpp"
#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
using PerfCountersTest = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(PerfCountersTest, NotEmptyWhenExecuted) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, std::string> config = {{ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
                                                   InferenceEngine::PluginConfigParams::YES }};
    config.insert(configuration.begin(), configuration.end());
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, config);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::ResponseDesc response;
    ASSERT_NO_FATAL_FAILURE(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer()) << response.msg;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    perfMap = req.GetPerformanceCounts();
    ASSERT_NE(perfMap.size(), 0);
}
}  // namespace BehaviorTestsDefinitions
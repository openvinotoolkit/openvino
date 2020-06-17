// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ie_preprocess.hpp"
#include "functional_test_utils/behavior_test_utils.hpp"
namespace BehaviorTestsDefinitions {
using VersionTest = BehaviorTestsUtils::BehaviorTestsBasic;

// Load unsupported network type to the Plugin
TEST_P(VersionTest, pluginCurrentVersionIsCorrect) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::string refError = "The plugin does not support";
    if (targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        std::map<std::string, InferenceEngine::Version> versions = ie->GetVersions(targetDevice);
        ASSERT_EQ(versions.size(), 1);
        auto version = versions.begin()->second;
        ASSERT_EQ(version.apiVersion.major, 2);
        ASSERT_EQ(version.apiVersion.minor, 1);
    }
}
}  // namespace BehaviorTestsDefinitions
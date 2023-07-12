// Copyright (C) 2018-2023 Intel Corporation
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
class VersionTest : public testing::WithParamInterface<std::string>,
                    public BehaviorTestsUtils::IEPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        std::string targetDevice;
        std::map<std::string, std::string> config;
        targetDevice = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void SetUp()  override {
        target_device = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
};

// Load unsupported network type to the Plugin
TEST_P(VersionTest, pluginCurrentVersionIsCorrect) {
    if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos) {
        std::map<std::string, InferenceEngine::Version> versions = ie->GetVersions(target_device);
        ASSERT_EQ(versions.size(), 1);
        ASSERT_EQ(versions.begin()->first, target_device);
        auto version = versions.begin()->second;
        IE_SUPPRESS_DEPRECATED_START
        ASSERT_EQ(version.apiVersion.major, 2);
        ASSERT_EQ(version.apiVersion.minor, 1);
        IE_SUPPRESS_DEPRECATED_END
    }
}
}  // namespace BehaviorTestsDefinitions

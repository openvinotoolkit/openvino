// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {
class VersionTests : public testing::WithParamInterface<std::string>, virtual public OVPluginTestBase {
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

    void SetUp() override {
        target_device = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
    }

    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
};

// Load unsupported network type to the Plugin
TEST_P(VersionTests, pluginCurrentVersionIsCorrect) {
    if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos) {
        std::map<std::string, ov::Version> versions = core->get_versions(target_device);
        ASSERT_EQ(versions.size(), 1);
        ASSERT_EQ(versions.begin()->first, target_device);
        auto version = versions.begin()->second;
        EXPECT_TRUE(version.buildNumber != nullptr);
        EXPECT_TRUE(version.description != nullptr);
    }
}
}  // namespace behavior
}  // namespace test
}  // namespace ov

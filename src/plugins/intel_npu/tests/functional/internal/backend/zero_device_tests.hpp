// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "zero_device.hpp"

namespace ov::test::behavior {

using namespace ::intel_npu;

using ZeroDeviceTestParams = std::tuple<std::string, ov::AnyMap>;

class ZeroDeviceTest : public ::testing::TestWithParam<ZeroDeviceTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ZeroDeviceTestParams>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice);
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "_configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(targetDevice, configuration) = this->GetParam();

        zeroInitStruct = ZeroInitStructsHolder::getInstance();
        ASSERT_NE(zeroInitStruct, nullptr);
        ASSERT_NE(zeroInitStruct->getDevice(), nullptr);

        device = std::make_unique<ZeroDevice>(zeroInitStruct);
    }

    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;
    std::unique_ptr<ZeroDevice> device;
};

// E2E tests fail earlier (at metadata parse), so this narrow unit test provides
// the only coverage for the L0 driver validation branch with invalid descriptors.
TEST_P(ZeroDeviceTest, ValidateRejectsInvalidString) {
    if (zeroInitStruct->getZeDrvApiVersion() < ZE_MAKE_VERSION(1, 16)) {
        ASSERT_ANY_THROW(device->validateCompatibilityDescriptor("not_a_valid_compat_string"));
    } else {
        bool isCompatible = true;
        OV_ASSERT_NO_THROW(isCompatible = device->validateCompatibilityDescriptor("not_a_valid_compat_string"));
        EXPECT_FALSE(isCompatible);
    }
}

}  // namespace ov::test::behavior

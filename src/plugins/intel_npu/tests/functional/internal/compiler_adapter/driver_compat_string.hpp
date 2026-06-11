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

#include "common/utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "driver_compiler_adapter.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace ov::test::behavior {

using namespace ::intel_npu;

using CompatStringParams = std::tuple<std::string, ov::AnyMap>;

class DriverCompatStringTest : public ::testing::TestWithParam<CompatStringParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompatStringParams>& obj) {
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

        zel_version_t loaderVer = {};
        zel_component_version_t cv = {};
        if (zelGetLoaderVersion(&cv) == ZE_RESULT_SUCCESS) {
            loaderVer = cv.component_lib_version;
        }
        if (loaderVer.major == 1 && loaderVer.minor < 29) {
            GTEST_SKIP() << "ze_loader version " << loaderVer.major << "." << loaderVer.minor
                         << " < 1.29: runtime requirements extension entry-points not forwarded";
        }
        if (ZeroApi::get_instance()->zeDeviceValidateRuntimeRequirements == nullptr) {
            GTEST_SKIP() << "Driver does not implement zeDeviceValidateRuntimeRequirements; "
                            "compatibility descriptor validation is unavailable";
        }
        adapter = std::make_unique<DriverCompilerAdapter>(zeroInitStruct);
    }

    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;
    std::unique_ptr<DriverCompilerAdapter> adapter;
};

// a E2E test fails earlier (metadata parse), so this narrow unit test is the only coverage for the L0 driver validation
// branch.
TEST_P(DriverCompatStringTest, ValidateRejectsGarbageString) {
    bool isCompatible = true;
    OV_ASSERT_NO_THROW(isCompatible = adapter->validate_compatibility_descriptor("not_a_valid_compat_string"));
    EXPECT_FALSE(isCompatible);
}

// no E2E test reaches this branch because compilation never produces an empty descriptor.
TEST_P(DriverCompatStringTest, ValidateRejectsEmptyString) {
    bool isCompatible = true;
    OV_ASSERT_NO_THROW(isCompatible = adapter->validate_compatibility_descriptor(""));
    EXPECT_FALSE(isCompatible);
}

}  // namespace ov::test::behavior

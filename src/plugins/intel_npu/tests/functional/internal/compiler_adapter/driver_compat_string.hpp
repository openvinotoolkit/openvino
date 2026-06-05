// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "compiler_adapter_utils.hpp"
#include "driver_compiler_adapter.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "ze_graph_ext_wrappers.hpp"

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

        zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);
        adapter = std::make_unique<DriverCompilerAdapter>(zeroInitStruct);
    }

    void TearDown() override {
        if (zeGraphExt && graphDescriptor._handle != nullptr) {
            zeGraphExt->destroyGraph(graphDescriptor);
        }
    }

    void compileModel() {
        auto model = ov::test::utils::make_multi_single_conv();
        graphDescriptor = zeGraphExt->getGraphDescriptor(makeTestSerializedIR(model, zeroInitStruct), "", false);
    }

    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;
    std::shared_ptr<ZeGraphExtWrappers> zeGraphExt;
    std::unique_ptr<DriverCompilerAdapter> adapter;
    GraphDescriptor graphDescriptor;
};

TEST_P(DriverCompatStringTest, GetRuntimeRequirements_ReturnsStringAfterCompile) {
    if (!adapter->is_option_supported(RUNTIME_REQUIREMENTS::key().data())) {
        GTEST_SKIP() << "RUNTIME_REQUIREMENTS not supported by this driver";
    }

    OV_ASSERT_NO_THROW(compileModel());
    ASSERT_NE(graphDescriptor._handle, nullptr);

    std::optional<std::string> result;
    OV_ASSERT_NO_THROW(result = adapter->get_runtime_requirements(graphDescriptor));
    if (!result.has_value()) {
        GTEST_SKIP() << "get_runtime_requirements returned nullopt (feature present but disabled in this driver)";
    }
    ASSERT_FALSE(result->empty()) << "get_runtime_requirements returned an empty string";
    std::cout << "[compat] requirements string: " << *result << "\n";
}

TEST_P(DriverCompatStringTest, ValidateCompatibilityDescriptor_AcceptsOwnString) {
    if (!adapter->is_option_supported(RUNTIME_REQUIREMENTS::key().data())) {
        GTEST_SKIP() << "RUNTIME_REQUIREMENTS not supported by this driver";
    }
    if (!adapter->is_option_supported(COMPATIBILITY_CHECK::key().data())) {
        GTEST_SKIP() << "COMPATIBILITY_CHECK not supported by this driver";
    }

    OV_ASSERT_NO_THROW(compileModel());
    ASSERT_NE(graphDescriptor._handle, nullptr);

    const auto compatStr = adapter->get_runtime_requirements(graphDescriptor);
    if (!compatStr.has_value()) {
        GTEST_SKIP() << "get_runtime_requirements returned nullopt (feature present but disabled in this driver)";
    }
    std::cout << "[compat] compat string: " << *compatStr << "\n";

    bool isCompatible = false;
    OV_ASSERT_NO_THROW(isCompatible = adapter->validate_compatibility_descriptor(*compatStr));
    EXPECT_TRUE(isCompatible) << "adapter rejected a compat string it just generated on the same device";
}

TEST_P(DriverCompatStringTest, ValidateCompatibilityDescriptor_RejectsGarbageString) {
    if (!adapter->is_option_supported(COMPATIBILITY_CHECK::key().data())) {
        GTEST_SKIP() << "COMPATIBILITY_CHECK not supported by this driver";
    }

    bool isCompatible = true;
    OV_ASSERT_NO_THROW(isCompatible = adapter->validate_compatibility_descriptor("not_a_valid_compat_string"));
    std::cout << "[compat] garbage string accepted: " << isCompatible << "\n";
    EXPECT_FALSE(isCompatible);
}

TEST_P(DriverCompatStringTest, ValidateCompatibilityDescriptor_RejectsEmptyString) {
    if (!adapter->is_option_supported(COMPATIBILITY_CHECK::key().data())) {
        GTEST_SKIP() << "COMPATIBILITY_CHECK not supported by this driver";
    }

    bool isCompatible = true;
    OV_ASSERT_NO_THROW(isCompatible = adapter->validate_compatibility_descriptor(""));
    std::cout << "[compat] empty string accepted: " << isCompatible << "\n";
    EXPECT_FALSE(isCompatible);
}

TEST_P(DriverCompatStringTest, IsOptionSupported_CompatibilityCheck) {
    const bool supported = adapter->is_option_supported(COMPATIBILITY_CHECK::key().data());
    std::cout << "[compat] COMPATIBILITY_CHECK supported: " << supported << "\n";
    EXPECT_EQ(supported, zeDeviceValidateRuntimeRequirements != nullptr);
}

TEST_P(DriverCompatStringTest, IsOptionSupported_RuntimeRequirements) {
    const bool supported = adapter->is_option_supported(RUNTIME_REQUIREMENTS::key().data());
    std::cout << "[compat] RUNTIME_REQUIREMENTS supported: " << supported << "\n";
    EXPECT_EQ(supported, zeDeviceGetRuntimeRequirements != nullptr);
}

TEST_P(DriverCompatStringTest, GetRuntimeRequirements_ReturnsNulloptWhenUnsupported) {
    OV_ASSERT_NO_THROW(compileModel());
    std::optional<std::string> result;
    OV_ASSERT_NO_THROW(result = adapter->get_runtime_requirements(graphDescriptor));
    std::cout << "[compat] get_runtime_requirements has_value: " << result.has_value() << "\n";
    if (result.has_value()) {
        GTEST_SKIP() << "Driver returned a compat string -- this test targets drivers where the feature is disabled";
    }
}

}  // namespace ov::test::behavior

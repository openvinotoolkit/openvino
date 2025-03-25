// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/npu_private_properties.hpp"

using CompilerType = ov::intel_npu::CompilerType;

namespace {

class TestCompiledModelNPU : public ov::test::behavior::OVPluginTestBase,
                             public testing::WithParamInterface<std::tuple<std::string, ov::AnyMap>> {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<std::string, ov::AnyMap>> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    ov::AnyMap configuration;
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
};

TEST_P(TestCompiledModelNPU, samePlatformProduceTheSameBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        std::string platform = ov::test::utils::getTestsPlatformFromEnvironmentOr("3720");

        configuration[ov::intel_npu::defer_weights_load.name()] = true;
        auto configuration1 = configuration;
        configuration1[ov::intel_npu::platform.name()] = platform;
        const auto& ov_model1 = buildSingleLayerSoftMaxNetwork();
        auto compiled_model1 = core->compile_model(ov_model1, target_device, configuration1);
        std::stringstream blobStream1;
        compiled_model1.export_model(blobStream1);

        auto configuration2 = configuration;
        configuration2[ov::intel_npu::platform.name()] = platform;
        const auto& ov_model2 = buildSingleLayerSoftMaxNetwork();
        auto compiled_model2 = core->compile_model(ov_model2, target_device, configuration2);
        std::stringstream blobStream2;
        compiled_model2.export_model(blobStream2);

        ASSERT_NE(0, blobStream1.str().size());
        ASSERT_EQ(0, std::memcmp(blobStream1.str().c_str(), blobStream2.str().c_str(), blobStream1.str().size()));
    }
}

class TestCompileModelWithoutDeviceNPU : public TestCompiledModelNPU {
protected:
    void SetUp() override {
        const auto devices = core->get_available_devices();
        const auto isNPUDeviceAvailable =
            std::find_if(devices.cbegin(), devices.cend(), [this](const std::string& device) {
                return device.find(target_device) != std::string::npos;
            }) != devices.cend();
        if (isNPUDeviceAvailable) {
            GTEST_SKIP() << "Skip the tests since device is available";
        }
    }
};

TEST_P(TestCompileModelWithoutDeviceNPU, ThrowIfNoDeviceAndNoPlatform) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& ov_model = buildSingleLayerSoftMaxNetwork();
        ASSERT_THROW(auto compiled_model = core->compile_model(ov_model, target_device, configuration), ov::Exception);
    }
}

TEST_P(TestCompileModelWithoutDeviceNPU, NoThrowIfNoDeviceAndButPlatformPassed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto netConfiguration = configuration;
        netConfiguration[ov::intel_npu::platform.name()] = ov::test::utils::getTestsPlatformFromEnvironmentOr("3720");
        const auto& ov_model = buildSingleLayerSoftMaxNetwork();
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_model, target_device, netConfiguration));
    }
}

const std::map<std::string_view, std::array<std::string_view, 2>> wrongDevice = {
    // {orig, {wrong for MLIR}}
    {"VPU3720", {"VPU0000"}},
};

std::string getWrongDevice(const std::string_view platform, const CompilerType&) {
    // here we mix up devices in order to test the check on the runtime side
    auto device = wrongDevice.find(platform);

    if (device == wrongDevice.end()) {
        OPENVINO_THROW("Cannot map wrong device for the platform ", platform);
    }
    return std::string(device->second[0]);
}

const std::map<std::string_view, std::array<std::string_view, 2>> validDevice = {
    // {orig, {valid for MLIR}}
    {"VPU3720", {"VPU3720"}}};

std::string getValidDevice(const std::string_view platform, const CompilerType&) {
    auto device = validDevice.find(platform);

    if (device == validDevice.end()) {
        OPENVINO_THROW("Cannot map valid device for the platform ", platform);
    }
    return std::string(device->second[0]);
}

TEST_P(TestCompileModelWithoutDeviceNPU, CheckDeviceInBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        // Compile model to target plugins, wrong platform specified -> expect an exception
        auto netConfigurationMLIR_wrong = configuration;
        netConfigurationMLIR_wrong[ov::intel_npu::platform.name()] =
            getWrongDevice(PlatformEnvironment::PLATFORM, CompilerType::MLIR);
        netConfigurationMLIR_wrong[ov::intel_npu::compiler_type.name()] = "MLIR";
        const auto& ov_model1 = buildSingleLayerSoftMaxNetwork();
        EXPECT_ANY_THROW(auto compiled_model =
                             core->compile_model(ov_model1, target_device, netConfigurationMLIR_wrong));

        // Compile model to target plugins, valid platform specified -> expect no exception
        auto netConfigurationMLIR_valid = configuration;
        netConfigurationMLIR_valid[ov::intel_npu::platform.name()] =
            getValidDevice(PlatformEnvironment::PLATFORM, CompilerType::MLIR);
        netConfigurationMLIR_valid[ov::intel_npu::compiler_type.name()] = "MLIR";
        const auto& ov_model2 = buildSingleLayerSoftMaxNetwork();
        EXPECT_NO_THROW(auto compiled_model =
                            core->compile_model(ov_model2, target_device, netConfigurationMLIR_valid));
    }
}

}  // namespace

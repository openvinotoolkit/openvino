// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "intel_npu/common/compiler_selection.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_backend.hpp"

namespace ov {
namespace test {
namespace behavior {

using CrossCompilationParams = std::tuple<std::string, ov::AnyMap>;

namespace {

std::string getTargetPlatform(const ov::AnyMap& config) {
    const auto platform = config.find(ov::intel_npu::platform.name());
    return platform == config.end() ? std::string{} : platform->second.as<std::string>();
}

ov::intel_npu::CompilerType getExpectedCompilerType(const std::string& targetPlatform,
                                                    const std::string& devicePlatform,
                                                    const std::shared_ptr<::intel_npu::IDevice>& device) {
    const auto canonicalDevicePlatform = ov::intel_npu::Platform::standardize(devicePlatform);
    const auto canonicalTargetPlatform = ov::intel_npu::Platform::standardize(targetPlatform);
    const auto resolvedTargetPlatform =
        (targetPlatform.empty() || targetPlatform == ov::intel_npu::Platform::AUTO_DETECT)
            ? canonicalDevicePlatform
            : canonicalTargetPlatform;
    const bool onlineCompilation = resolvedTargetPlatform == canonicalDevicePlatform;

    if (!onlineCompilation) {
        return ov::intel_npu::CompilerType::PLUGIN;
    }

#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr bool windowsDriver = true;
#else
    constexpr bool windowsDriver = false;
#endif
    std::optional<uint32_t> driverVersion;
    if (resolvedTargetPlatform == ov::intel_npu::Platform::NPU3720) {
        driverVersion = device->getDeviceProperties().driverVersion;
    }

    return ::intel_npu::isPluginCompilerPreferred(resolvedTargetPlatform, driverVersion, windowsDriver)
               ? ov::intel_npu::CompilerType::PLUGIN
               : ov::intel_npu::CompilerType::DRIVER;
}

}  // namespace

class CrossCompilationNPU : public OVPluginTestBase, public testing::WithParamInterface<CrossCompilationParams> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(target_device, config) = this->GetParam();

        model = ov::test::utils::make_multi_single_conv();

        APIBaseTest::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CrossCompilationParams>& obj) {
        std::string target_device;
        ov::AnyMap config;
        std::tie(target_device, config) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << target_device << "_";

        if (!config.empty()) {
            for (auto& configItem : config) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void TearDown() override {
        APIBaseTest::TearDown();
    }

protected:
    ov::Core core;
    ov::AnyMap config;
    std::shared_ptr<ov::Model> model;
};

TEST_P(CrossCompilationNPU, CrossCompilationTest) {
    auto backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    const auto device = backend->getDevice();
    ASSERT_NE(device, nullptr);
    if (!device) {
        return;
    }
    auto device_platform = device->getName();
    auto target_platform = getTargetPlatform(config);

    if (target_platform.empty() ||
        ov::intel_npu::Platform::standardize(device_platform) == ov::intel_npu::Platform::standardize(target_platform)) {
        GTEST_SKIP() << "Target platform resolves to the online device platform.";
    }

    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, target_device, config));

    ov::intel_npu::CompilerType compiler_type = ov::intel_npu::CompilerType::PREFER_PLUGIN;
    OV_ASSERT_NO_THROW(
        compiler_type =
            compiledModel.get_property(ov::intel_npu::compiler_type.name()).as<ov::intel_npu::CompilerType>());

    EXPECT_EQ(compiler_type, getExpectedCompilerType(target_platform, device_platform, device));
}

TEST_P(CrossCompilationNPU, OnlineCompilationTest) {
    auto backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    const auto device = backend->getDevice();
    ASSERT_NE(device, nullptr);
    if (!device) {
        return;
    }
    auto device_platform = device->getName();
    auto target_platform = getTargetPlatform(config);

    if (!target_platform.empty() && target_platform != ov::intel_npu::Platform::AUTO_DETECT &&
        ov::intel_npu::Platform::standardize(device_platform) !=
            ov::intel_npu::Platform::standardize(target_platform)) {
        GTEST_SKIP() << "Device platform does not match target platform.";
    }

    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, target_device, config));

    ov::intel_npu::CompilerType compiler_type = ov::intel_npu::CompilerType::PREFER_PLUGIN;
    OV_ASSERT_NO_THROW(
        compiler_type =
            compiledModel.get_property(ov::intel_npu::compiler_type.name()).as<ov::intel_npu::CompilerType>());

    EXPECT_EQ(compiler_type, getExpectedCompilerType(target_platform, device_platform, device));
}

const std::vector<ov::AnyMap> config = {{},
                                        {ov::intel_npu::platform(ov::intel_npu::Platform::AUTO_DETECT)},
                                        {ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720)},
                                        {ov::intel_npu::platform("NPU3720")},
                                        {ov::intel_npu::platform("VPU3720")},
                                        {ov::intel_npu::platform(ov::intel_npu::Platform::NPU4000)},
                                        {ov::intel_npu::platform(ov::intel_npu::Platform::NPU5010)},
                                        {ov::intel_npu::platform(ov::intel_npu::Platform::NPU5020)},
                                        {ov::intel_npu::platform(ov::intel_npu::Platform::NPU6010)}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         CrossCompilationNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(config)),
                         ov::test::utils::appendPlatformTypeTestName<CrossCompilationNPU>);

}  // namespace behavior
}  // namespace test
}  // namespace ov

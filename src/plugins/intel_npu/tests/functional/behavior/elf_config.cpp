// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <base/ov_behavior_test_utils.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "npu_private_properties.hpp"

namespace {

class ElfConfigTests :
        public ov::test::behavior::OVPluginTestBase,
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

TEST_P(ElfConfigTests, CompilationWithSpecificConfig) {
    if (getBackendName(*core) == "LEVEL0") {
        GTEST_SKIP() << "Skip due to failure on device";
    }
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& ov_model = buildSingleLayerSoftMaxNetwork();
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_model, target_device, configuration));
    }
}

const std::vector<ov::AnyMap> configs = {
        {{ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720)},
         {ov::intel_npu::use_elf_compiler_backend(ov::intel_npu::ElfCompilerBackend::NO)}}};

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {{ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720)},
         {ov::intel_npu::use_elf_compiler_backend(ov::intel_npu::ElfCompilerBackend::NO)},
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_ELF, ElfConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ElfConfigTests::getTestCaseName);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_ELF_Driver, ElfConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ElfConfigTests::getTestCaseName);
}  // namespace

//
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

class CompileForDifferentPlatformsTests :
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

// [Track number: E#15711]
// [Track number: E#15635]
TEST_P(CompileForDifferentPlatformsTests, CompilationForSpecificPlatform) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg["NPU_COMPILER_TYPE"] = "MLIR";
        cfg["NPU_CREATE_EXECUTOR"] = "0";
        const auto& ov_model = buildSingleLayerSoftMaxNetwork();
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_model, target_device, cfg));
    }
}

const std::vector<ov::AnyMap> configs = {
        {{ov::intel_npu::platform(ov::intel_npu::Platform::NPU3700)},
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR)},
        {{ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720)},
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR)},
        {{ov::device::id("3700")}, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR)},
        {{ov::device::id("3720")}, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR)}
};

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)},
        {{ov::intel_npu::platform(ov::intel_npu::Platform::NPU3700)},
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)},
        {{ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720)},
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)},
        {{ov::device::id("3700")}, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)},
        {{ov::device::id("3720")}, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, CompileForDifferentPlatformsTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         CompileForDifferentPlatformsTests::getTestCaseName);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_Driver, CompileForDifferentPlatformsTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         CompileForDifferentPlatformsTests::getTestCaseName);
}  // namespace

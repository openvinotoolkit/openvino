// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/config/common.hpp"
#include "shared_test_classes/subgraph/split_conv_concat.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov::test::behavior {

class DriverCompilerAdapterPropComTestNPU : public ov::test::behavior::OVPluginTestBase,
                                            public testing::WithParamInterface<CompilationParams> {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

protected:
    ov::AnyMap configuration;
};

TEST_P(DriverCompilerAdapterPropComTestNPU, TestNewPro) {
    auto simpleFunc = ov::test::utils::make_split_conv_concat();
    ov::Core core;
    EXPECT_NO_THROW(auto model = core.compile_model(simpleFunc, target_device, configuration));
}

const std::vector<ov::AnyMap> configs = {
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
     ov::intel_npu::compilation_mode_params("dummy-op-replacement=true")},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
     ov::intel_npu::compilation_mode_params("dummy-op-replacement=true optimization-level=1")},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
     ov::intel_npu::compilation_mode_params("optimization-level=1 dummy-op-replacement=true")},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
     ov::intel_npu::compilation_mode_params("optimization-level=1")},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
     ov::intel_npu::compilation_mode_params("optimization-level=1 performance-hint-override=latency")},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
     ov::intel_npu::compilation_mode_params("performance-hint-override=latency")}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DriverCompilerAdapterPropComTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         DriverCompilerAdapterPropComTestNPU::getTestCaseName);
}  // namespace ov::test::behavior

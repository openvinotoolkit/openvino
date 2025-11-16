// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include <random>

#include "common/functions.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/opsets/opset11.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "vcl_serializer.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov::test::behavior {

class DriverCompilerAdapterCustomStreamTestNPU : public ov::test::behavior::OVPluginTestBase,
                                                 public testing::WithParamInterface<CompilationParams> {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
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

TEST_P(DriverCompilerAdapterCustomStreamTestNPU, TestLargeModelWeightsCopy) {
    auto model = createModelWithLargeSize();
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};
    EXPECT_NO_THROW(::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, true));
}

TEST_P(DriverCompilerAdapterCustomStreamTestNPU, TestLargeModelNoWeightsCopy) {
    auto model = createModelWithLargeSize();
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};

    EXPECT_NO_THROW(::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false, 0));
    EXPECT_NO_THROW(::intel_npu::driver_compiler_utils::serializeIR(model, dummyCompilerVersion, 11, false, 100));
    EXPECT_NO_THROW(::intel_npu::driver_compiler_utils::serializeIR(model,
                                                                    dummyCompilerVersion,
                                                                    11,
                                                                    false,
                                                                    static_cast<size_t>(1e9)));
}

const std::vector<ov::AnyMap> configs = {
    {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DriverCompilerAdapterCustomStreamTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         DriverCompilerAdapterCustomStreamTestNPU::getTestCaseName);
}  // namespace ov::test::behavior

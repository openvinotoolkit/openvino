// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "overload/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;

namespace {

static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string target_device = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    return "target_device=" + target_device +
           "_targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU) + "_";
}

// MLIR compiler type config
const std::vector<ov::AnyMap> mlirCompilerConfigs = {{
        ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
        ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
}};

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

const std::vector<std::string> device_names_and_priorities = {
        "MULTI:NPU",  // NPU via MULTI,
        "AUTO:NPU",   // NPU via AUTO,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTest, ::testing::Values(ov::test::utils::DEVICE_NPU),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::utils::DEVICE_NPU), getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(mlirCompilerConfigs)),
                         OVHoldersTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestOnImportedNetworkNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(mlirCompilerConfigs)),
                         OVHoldersTestOnImportedNetworkNPU::getTestCaseName);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVHoldersTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         OVHoldersTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVHoldersTestOnImportedNetworkNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         OVHoldersTestOnImportedNetworkNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests, OVHoldersTestWithConfig,
                         ::testing::ValuesIn(device_names_and_priorities),
                         (ov::test::utils::appendPlatformTypeTestName<OVHoldersTestWithConfig>));
}  // namespace

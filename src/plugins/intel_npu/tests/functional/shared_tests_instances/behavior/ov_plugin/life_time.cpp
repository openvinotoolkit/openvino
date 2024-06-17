// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "overload/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;

namespace {

static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string target_device = obj.param;
    return "target_device=" + ov::test::utils::getTestsDeviceNameFromEnvironmentOr(target_device) + "_" +
           "targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU) + "_";
}

const std::vector<ov::AnyMap> configs = {{}};

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
                                            ::testing::ValuesIn(configs)),
                         OVHoldersTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestOnImportedNetworkNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         OVHoldersTestOnImportedNetworkNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests, OVHoldersTestWithConfig,
                         ::testing::ValuesIn(device_names_and_priorities),
                         (ov::test::utils::appendPlatformTypeTestName<OVHoldersTestWithConfig>));
}  // namespace

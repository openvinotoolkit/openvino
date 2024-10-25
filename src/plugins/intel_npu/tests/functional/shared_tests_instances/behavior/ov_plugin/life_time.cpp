// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"
#include "overload/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;

namespace {

static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string target_device = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    return "target_device=" + target_device +
           "_targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU) + "_";
}

const std::vector<ov::AnyMap> configs = {{}};

const std::vector<std::string> device_names_and_priorities = {
    "MULTI:NPU",  // NPU via MULTI,
    "AUTO:NPU",   // NPU via AUTO,
};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVHoldersTest,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVHoldersTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         OVHoldersTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVHoldersTestOnImportedNetworkNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         OVHoldersTestOnImportedNetworkNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_VirtualPlugin_BehaviorTests,
                         OVHoldersTestWithConfig,
                         ::testing::ValuesIn(device_names_and_priorities),
                         (ov::test::utils::appendPlatformTypeTestName<OVHoldersTestWithConfig>));
}  // namespace

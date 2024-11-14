// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/wait.hpp"

#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<ov::AnyMap> multiConfigs = {
    {ov::device::priorities(ov::test::utils::DEVICE_NPU), ov::device::properties(ov::test::utils::DEVICE_NPU, {})}};

const std::vector<ov::AnyMap> autoConfigs = {
    {ov::device::priorities(ov::test::utils::DEVICE_NPU), ov::device::properties(ov::test::utils::DEVICE_NPU, {})}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestWaitTests>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Multi_BehaviorTests,
                         OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestWaitTests>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Auto_BehaviorTests,
                         OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestWaitTests>);

}  // namespace

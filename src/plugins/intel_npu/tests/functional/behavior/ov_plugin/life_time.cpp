// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "life_time.hpp"

#include "common/utils.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVHoldersTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVHoldersTestNPU>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVHoldersTestOnImportedNetworkNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVHoldersTestOnImportedNetworkNPU>);

}  // namespace

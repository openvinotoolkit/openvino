// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/version.hpp"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"

namespace ov {
namespace test {
namespace behavior {

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests, VersionTests, ::testing::Values(ov::test::utils::DEVICE_NPU),
                         (ov::test::utils::appendPlatformTypeTestName<VersionTests>));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Hetero_BehaviorTests, VersionTests, ::testing::Values(ov::test::utils::DEVICE_HETERO),
                         (ov::test::utils::appendPlatformTypeTestName<VersionTests>));

}  // namespace behavior
}  // namespace test
}  // namespace ov

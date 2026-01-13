// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/query_model.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"

namespace ov {
namespace test {
namespace behavior {

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_OVClassModelTestP,
                         OVClassModelTestP,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         (ov::test::utils::appendPlatformTypeTestName<OVClassModelTestP>));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_OVClassQueryModelTestTests,
                         OVClassQueryModelTest,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         (ov::test::utils::appendPlatformTypeTestName<OVClassQueryModelTest>));

}  // namespace behavior
}  // namespace test
}  // namespace ov

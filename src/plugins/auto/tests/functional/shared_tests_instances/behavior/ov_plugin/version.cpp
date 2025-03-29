// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/version.hpp"

namespace ov {
namespace test {
namespace behavior {

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         VersionTests,
                         ::testing::Values(ov::test::utils::DEVICE_MULTI),
                         VersionTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         VersionTests,
                         ::testing::Values(ov::test::utils::DEVICE_AUTO),
                         VersionTests::getTestCaseName);

}  // namespace behavior
}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/version.hpp"

namespace ov {
namespace test {
namespace behavior {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         VersionTests,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         VersionTests::getTestCaseName);

}  // namespace behavior
}  // namespace test
}  // namespace ov

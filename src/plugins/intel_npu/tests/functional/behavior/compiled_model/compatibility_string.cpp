// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/npu_test_env_cfg.hpp"
#include "compatibility_string.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         ClassCompatibilityStringTestSuite,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ClassCompatibilityStringTestSuite::getTestCaseName);
} // namespace

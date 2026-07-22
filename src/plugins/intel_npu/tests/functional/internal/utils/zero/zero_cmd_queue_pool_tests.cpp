// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/utils/zero/zero_cmd_queue_pool_tests.hpp"

#include "common/utils.hpp"

using namespace ov::test::behavior;

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         ZeroCmdQueuePoolTests,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ZeroCmdQueuePoolTests::getTestCaseName);

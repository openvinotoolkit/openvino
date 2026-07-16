// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_allocator_test.hpp"

#include "common/npu_test_env_cfg.hpp"

namespace {

using ov::test::behavior::VclAllocatorFuncTests;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         VclAllocatorFuncTests,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         VclAllocatorFuncTests::getTestCaseName);
}  // namespace

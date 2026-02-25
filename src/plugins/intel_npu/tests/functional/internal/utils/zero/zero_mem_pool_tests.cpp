// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/utils/zero/zero_mem_pool_tests.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;


INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         ZeroMemPoolTests,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ZeroMemPoolTests::getTestCaseName);

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/work_with_devices.hpp"

#include "common/utils.hpp"

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         TestCompiledModelNPU,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         TestCompiledModelNPU::getTestCaseName);

}  // namespace

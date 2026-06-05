// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compat_string.hpp"

namespace {
using ov::test::behavior::DriverCompatStringTest;
const std::vector<ov::AnyMap> emptyConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DriverCompatStringTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfigs)),
                         DriverCompatStringTest::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_device_tests.hpp"

namespace {
using namespace ov::test::behavior;

const std::vector<ov::AnyMap> emptyConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         ZeroDeviceTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfigs)),
                         ZeroDeviceTest::getTestCaseName);
}  // namespace

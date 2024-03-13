// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/ov_plugin/core_threading.hpp>
#include "behavior/ov_plugin/core_threading.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_plugin, CoreThreadingTest,
                         testing::Values(std::tuple<Device, Config>{ov::test::utils::target_device, {{ov::enable_profiling(false)}}}),
                         CoreThreadingTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::Values(std::tuple<Device, Config>{ov::test::utils::target_device, {{ov::enable_profiling(false)}}}),
                                          testing::Values(4),
                                          testing::Values(50)),
                         CoreThreadingTestsWithIter::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, CoreThreadingTestsWithCacheEnabled,
    testing::Combine(testing::Values(std::tuple<Device, Config>{ov::test::utils::target_device, {{ov::enable_profiling(false)}}}),
                     testing::Values(20),
                     testing::Values(10)),
    CoreThreadingTestsWithCacheEnabled::getTestCaseName);

} // namespace
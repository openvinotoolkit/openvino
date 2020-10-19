// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/core_threading_tests.hpp>

namespace {

Params params[] = {
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_GNA, {{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_HETERO, {{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_GNA }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MULTI, {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_GNA }}},
};

}  // namespace

INSTANTIATE_TEST_CASE_P(GNA, CoreThreadingTests, testing::ValuesIn(params), CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(DISABLED_GNA, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(2),
                     testing::Values(2)),
    CoreThreadingTestsWithIterations::getTestCaseName);

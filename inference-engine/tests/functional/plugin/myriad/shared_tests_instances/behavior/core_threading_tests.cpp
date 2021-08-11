// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/core_threading_tests.hpp>

namespace {

Params params[] = {
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MYRIAD, {{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_HETERO, {{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_MYRIAD }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MULTI, {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_MYRIAD }}},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(MYRIAD, CoreThreadingTests, testing::ValuesIn(params), CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_MYRIAD, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(2),
                     testing::Values(2),
                     testing::Values(ModelClass::Default)),
    CoreThreadingTestsWithIterations::getTestCaseName);

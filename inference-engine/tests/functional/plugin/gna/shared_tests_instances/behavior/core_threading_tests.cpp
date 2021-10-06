// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/core_threading_tests.hpp>

namespace {
Params params[] = {
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_GNA, {{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_HETERO, {{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_GNA }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MULTI, {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_GNA }}},
};
// TODO: Consider to append params[1] after issue *-45658 resolved
std::vector< std::tuple<Device, Config> > paramsWithIterations{ params[0], params[2] };
}  // namespace

INSTANTIATE_TEST_SUITE_P(GNA, CoreThreadingTests, testing::ValuesIn(params), CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GNA, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(paramsWithIterations),
                     testing::Values(3),
                     testing::Values(4),
                     testing::Values(ModelClass::ConvPoolRelu)),
    CoreThreadingTestsWithIterations::getTestCaseName);

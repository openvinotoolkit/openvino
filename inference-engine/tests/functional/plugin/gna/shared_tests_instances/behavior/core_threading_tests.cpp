// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/core_threading_tests.hpp>

namespace {
std::pair<std::string, std::string> nonDefaultModel{ "MODEL_TYPE", "ConvPoolRelu" };
Params params[] = {
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_GNA, {{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) },
        nonDefaultModel}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_HETERO, {{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_GNA },
        nonDefaultModel}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MULTI, {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_GNA },
        nonDefaultModel}},
};

}  // namespace

INSTANTIATE_TEST_CASE_P(GNA, CoreThreadingTests, testing::ValuesIn(params), CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GNA, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(3),
                     testing::Values(4)),
    CoreThreadingTestsWithIterations::getTestCaseName);

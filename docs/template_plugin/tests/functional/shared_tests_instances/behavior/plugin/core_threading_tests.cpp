// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/plugin/core_threading.hpp>

namespace {

const Params params[] = {
    std::tuple<Device, Config>{CommonTestUtils::DEVICE_TEMPLATE, {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}}},
    std::tuple<Device, Config>{CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK", CommonTestUtils::DEVICE_TEMPLATE}}},
    std::tuple<Device, Config>{CommonTestUtils::DEVICE_MULTI,
                               {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_TEMPLATE}}},
    std::tuple<Device, Config>{CommonTestUtils::DEVICE_AUTO,
                               {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_TEMPLATE}}},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(TEMPLATE, CoreThreadingTests, testing::ValuesIn(params), CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TEMPLATE,
                         CoreThreadingTestsWithIterations,
                         testing::Combine(testing::ValuesIn(params),
                                          testing::Values(4),
                                          testing::Values(50),
                                          testing::Values(ModelClass::Default)),
                         CoreThreadingTestsWithIterations::getTestCaseName);

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/plugin/core_threading.hpp>
#include "api_conformance_helpers.hpp"

using namespace ov::test::conformance;

namespace {

const Params coreThreadingParams[] = {
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_HETERO, generate_configs(CommonTestUtils::DEVICE_HETERO).front() },
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MULTI, generate_configs(CommonTestUtils::DEVICE_MULTI).front() },
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_AUTO, generate_configs(CommonTestUtils::DEVICE_AUTO).front() },
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_BATCH, generate_configs(CommonTestUtils::DEVICE_BATCH).front() },
};

INSTANTIATE_TEST_SUITE_P(ie_plugin_, CoreThreadingTests, testing::ValuesIn(coreThreadingParams), CoreThreadingTests::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(ie_plugin, CoreThreadingTests,
        ::testing::Combine(
                ::testing::ValuesIn(return_all_possible_device_combination()),
                ::testing::Values(Config{{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }})),
        CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ie_plugin, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(coreThreadingParams),
                     testing::Values(4),
                     testing::Values(50),
                     testing::Values(ModelClass::Default)),
    CoreThreadingTestsWithIterations::getTestCaseName);

}  // namespace

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/plugin/core_threading.hpp>
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;

std::string getDeviceName() {
    return ConformanceTests::targetDevice;
}

const Params params[] = {
    std::tuple<Device, Config>{ getDeviceName(), {{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_HETERO, generateConfigs(CommonTestUtils::DEVICE_HETERO).front() },
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MULTI, generateConfigs(CommonTestUtils::DEVICE_MULTI).front() },
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_AUTO, generateConfigs(CommonTestUtils::DEVICE_AUTO).front() },
};

INSTANTIATE_TEST_SUITE_P(Conformance, CoreThreadingTests, testing::ValuesIn(params), CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conformance, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(4),
                     testing::Values(50),
                     testing::Values(ModelClass::Default)),
    CoreThreadingTestsWithIterations::getTestCaseName);

}  // namespace

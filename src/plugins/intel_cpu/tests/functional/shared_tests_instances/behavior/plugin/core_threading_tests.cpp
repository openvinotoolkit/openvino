// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/plugin/core_threading.hpp>
#ifdef __GLIBC__
#include <gnu/libc-version.h>
#if __GLIBC_MINOR__  >= 34
    #define ENABLETESTMULTI
#endif
#endif

namespace {

const Params params[] = {
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_CPU, {{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_HETERO, {{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_CPU }}},
#ifdef ENABLETESTMULTI
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MULTI, {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_CPU }}},
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_AUTO, {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_CPU }}},
#endif
};

const Params paramsStreams[] = {
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_CPU, {{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS), CONFIG_VALUE(CPU_THROUGHPUT_AUTO) }}},
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(CPU, CoreThreadingTests, testing::ValuesIn(params), CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(4),
                     testing::Values(50),
                     testing::Values(ModelClass::Default)),
    CoreThreadingTestsWithIterations::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU_Streams, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(paramsStreams),
                     testing::Values(4),
                     testing::Values(50),
                     testing::Values(ModelClass::Default)),
    CoreThreadingTestsWithIterations::getTestCaseName);

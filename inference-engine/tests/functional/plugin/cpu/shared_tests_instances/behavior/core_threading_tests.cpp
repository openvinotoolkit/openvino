// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/core_threading_tests.hpp>

namespace {

const Params params[] = {
    std::tuple<Device, Config> { "CPU", { { CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) } } },
    std::tuple<Device, Config> { "HETERO", { { "TARGET_FALLBACK", "CPU" } } },
    std::tuple<Device, Config> { "MULTI", { { MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , "CPU" } } }
};

const Params paramsStreams[] = {
    std::tuple<Device, Config> { "CPU", { { CONFIG_KEY(CPU_THROUGHPUT_STREAMS), CONFIG_VALUE(CPU_THROUGHPUT_AUTO) } } }
};


}  // namespace

INSTANTIATE_TEST_CASE_P(CPU, CoreThreadingTests, testing::ValuesIn(params));

INSTANTIATE_TEST_CASE_P(CPU, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(4),
                     testing::Values(50)));

INSTANTIATE_TEST_CASE_P(CPU_Streams, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(paramsStreams),
                     testing::Values(4),
                     testing::Values(10)));

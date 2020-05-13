// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/core_threading_tests.hpp>

namespace {

Params params[] = {
    std::tuple<Device, Config> { "CPU", { { CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) } } },
    std::tuple<Device, Config> { "HETERO", { { "TARGET_FALLBACK", "CPU" } } },
    std::tuple<Device, Config> { "MULTI", { { MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , "CPU" } } }
};

}  // namespace

INSTANTIATE_TEST_CASE_P(CPU, CoreThreadingTests, testing::ValuesIn(params));

INSTANTIATE_TEST_CASE_P(CPU, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(4),
                     testing::Values(50)));

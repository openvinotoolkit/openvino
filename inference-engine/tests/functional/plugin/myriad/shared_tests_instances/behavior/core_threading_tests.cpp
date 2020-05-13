// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/core_threading_tests.hpp>

namespace {

Params params[] = {
    std::tuple<Device, Config> { "MYRIAD", { { CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) } } },
    std::tuple<Device, Config> { "HETERO", { { "TARGET_FALLBACK", "MYRIAD" } } },
    std::tuple<Device, Config> { "MULTI", { { MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , "MYRIAD" } } }
};

}  // namespace

INSTANTIATE_TEST_CASE_P(MYRIAD, CoreThreadingTests, testing::ValuesIn(params));

INSTANTIATE_TEST_CASE_P(DISABLED_MYRIAD, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(2),
                     testing::Values(2)));

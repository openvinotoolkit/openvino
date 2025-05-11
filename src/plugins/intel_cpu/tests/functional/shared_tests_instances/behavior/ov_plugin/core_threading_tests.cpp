// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_threading.hpp"

namespace {
const Params params[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_CPU, {{ov::enable_profiling(true)}}},
};

const Params paramsStreams[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_CPU, {{ov::num_streams(ov::streams::AUTO)}}},
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(CPU, CoreThreadingTest, testing::ValuesIn(params), CoreThreadingTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(params), testing::Values(4), testing::Values(50)),
                         CoreThreadingTestsWithIter::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU_Streams,
                         CoreThreadingTestsWithCacheEnabled,
                         testing::Combine(testing::ValuesIn(paramsStreams), testing::Values(20), testing::Values(10)),
                         CoreThreadingTestsWithCacheEnabled::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU_Streams,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(paramsStreams), testing::Values(4), testing::Values(50)),
                         CoreThreadingTestsWithIter::getTestCaseName);
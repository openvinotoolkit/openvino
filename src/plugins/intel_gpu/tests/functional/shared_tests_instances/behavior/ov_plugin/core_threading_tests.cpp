// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_threading.hpp"

namespace {
const Params params[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_GPU, {{ov::enable_profiling(true)}}},
};

const Params params_streams[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_GPU, {{ov::num_streams(ov::streams::AUTO)}}},
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(GPU, CoreThreadingTest, testing::ValuesIn(params), CoreThreadingTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GPU,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(params), testing::Values(4), testing::Values(50)),
                         CoreThreadingTestsWithIter::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GPU,
                         CoreThreadingTestsWithCacheEnabled,
                         testing::Combine(testing::ValuesIn(params_streams), testing::Values(20), testing::Values(10)),
                         CoreThreadingTestsWithCacheEnabled::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GPU_Streams,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(params_streams), testing::Values(4), testing::Values(50)),
                         CoreThreadingTestsWithIter::getTestCaseName);

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_threading.hpp"

namespace {
const Params params[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_HETERO,
                               {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}}},
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(nightly_HETERO,
                         CoreThreadingTest,
                         testing::ValuesIn(params),
                         CoreThreadingTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(HETERO_Streams,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(params), testing::Values(4), testing::Values(50)),
                         CoreThreadingTestsWithIter::getTestCaseName);

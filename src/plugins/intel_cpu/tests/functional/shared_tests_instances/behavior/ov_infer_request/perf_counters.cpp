// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {
        {}
};

const std::vector<ov::AnyMap> Multiconfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU)}
};

const std::vector<ov::AnyMap> Autoconfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU)}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                ::testing::ValuesIn(Multiconfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);
}  // namespace

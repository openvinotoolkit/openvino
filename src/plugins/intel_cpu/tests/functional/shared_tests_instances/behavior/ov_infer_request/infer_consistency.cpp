// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "behavior/ov_infer_request/infer_consistency.hpp"

using namespace ov::test::behavior;

namespace {
// for deviceConfigs, the deviceConfigs[0] is target device which need to be tested.
// deviceConfigs[1], deviceConfigs[2],deviceConfigs[n] are the devices which will
// be compared with target device, the result of target should be in one of the compared
// device.
using Configs = std::vector<std::pair<std::string, ov::AnyMap>>;

std::vector<Configs> configs = {
    {{ov::test::utils::DEVICE_CPU, {}}, {ov::test::utils::DEVICE_CPU, {}}}
};

std::vector<Configs> AutoConfigs = {
    {
        {
            ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_CPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
        },
        {ov::test::utils::DEVICE_CPU, {}}
    },
    {
        {
            ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_CPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}
        },
        {ov::test::utils::DEVICE_CPU, {}}
    },
    {
        {
            ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_CPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}
        },
        {ov::test::utils::DEVICE_CPU, {}}
    }
};

INSTANTIATE_TEST_SUITE_P(BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(10),// inferRequest num
        ::testing::Values(10),// infer counts
        ::testing::ValuesIn(configs)),
    OVInferConsistencyTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Auto_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(10),// inferRequest num
        ::testing::Values(10),// infer counts
        ::testing::ValuesIn(AutoConfigs)),
    OVInferConsistencyTest::getTestCaseName);
}  // namespace

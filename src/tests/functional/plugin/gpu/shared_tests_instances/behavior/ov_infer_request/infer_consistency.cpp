// Copyright (C) 2018-2022 Intel Corporation
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
    {{CommonTestUtils::DEVICE_GPU, {}}, {CommonTestUtils::DEVICE_GPU, {}}}
};

std::vector<Configs> AutoConfigs = {
    {
        {
            CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
        },
        {CommonTestUtils::DEVICE_GPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}
        },
        {CommonTestUtils::DEVICE_GPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}
        },
        {CommonTestUtils::DEVICE_GPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU +
            "," + CommonTestUtils::DEVICE_CPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
        },
        {CommonTestUtils::DEVICE_GPU, {}},  {CommonTestUtils::DEVICE_CPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU +
            "," + CommonTestUtils::DEVICE_CPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}
        },
        {CommonTestUtils::DEVICE_GPU, {}},  {CommonTestUtils::DEVICE_CPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU +
            "," + CommonTestUtils::DEVICE_CPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}
        },
        {CommonTestUtils::DEVICE_GPU, {}},  {CommonTestUtils::DEVICE_CPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_CPU +
            "," + CommonTestUtils::DEVICE_GPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}
        },
        {CommonTestUtils::DEVICE_CPU, {}},  {CommonTestUtils::DEVICE_GPU, {}}
    }
};

std::vector<Configs> MultiConfigs = {
    {
        {
            CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
        },
        {CommonTestUtils::DEVICE_GPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}
        },
        {CommonTestUtils::DEVICE_GPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU +
            "," + CommonTestUtils::DEVICE_CPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
        },
        {CommonTestUtils::DEVICE_GPU, {}},  {CommonTestUtils::DEVICE_CPU, {}}
    },
    {
        {
            CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU +
            "," + CommonTestUtils::DEVICE_CPU,
            {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}
        },
        {CommonTestUtils::DEVICE_GPU, {}},  {CommonTestUtils::DEVICE_CPU, {}}
    }
};



INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(10),// inferRequest num
        ::testing::Values(50),// infer counts
        ::testing::ValuesIn(configs)),
    OVInferConsistencyTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(10),// inferRequest num
        ::testing::Values(50),// infer counts
        ::testing::ValuesIn(AutoConfigs)),
    OVInferConsistencyTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(10),// inferRequest num
        ::testing::Values(50),// infer counts
        ::testing::ValuesIn(MultiConfigs)),
    OVInferConsistencyTest::getTestCaseName);
}  // namespace
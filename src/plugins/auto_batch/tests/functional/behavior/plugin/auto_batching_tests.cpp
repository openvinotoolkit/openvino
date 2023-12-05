// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "auto_batching_tests.hpp"

#include "behavior/plugin/configuration_tests.hpp"
#include "openvino/runtime/properties.hpp"

using namespace AutoBatchingTests;
using namespace BehaviorTestsDefinitions;

namespace {
const std::vector<bool> get_vs_set{true, false};
const std::vector<size_t> num_streams{1, 2};
const std::vector<size_t> num_requests{1, 3, 8, 9, 16, 64};
const std::vector<size_t> num_batch{1, 4, 8, 16, 32, 64, 128, 256};
INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_test,
                         AutoBatching_Test,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(num_streams),
                                            ::testing::ValuesIn(num_requests),
                                            ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_test,
                         AutoBatching_Test_DetectionOutput,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(num_streams),
                                            ::testing::ValuesIn(num_requests),
                                            ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test_DetectionOutput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_test,
                         DefaultConfigurationTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_BATCH) + ":" +
                                                              ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::Values(DefaultParameter{CONFIG_KEY(AUTO_BATCH_TIMEOUT),
                                                                               InferenceEngine::Parameter{"1000"}})),
                         DefaultConfigurationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_test_string,
                         DefaultConfigurationTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_BATCH) + ":" +
                                                              ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::Values(DefaultParameter{ov::auto_batch_timeout.name(),
                                                                               InferenceEngine::Parameter{"1000"}})),
                         DefaultConfigurationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_test_uint,
                         DefaultConfigurationTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_BATCH) + ":" +
                                                              ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::Values(DefaultParameter{ov::auto_batch_timeout.name(),
                                                                               InferenceEngine::Parameter{uint32_t(1000)}})),
                         DefaultConfigurationTest::getTestCaseName);

}  // namespace

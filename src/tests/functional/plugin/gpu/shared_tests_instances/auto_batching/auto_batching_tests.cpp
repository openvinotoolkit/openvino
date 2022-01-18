// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <auto_batching/auto_batching_tests.hpp>

const std::vector<size_t> num_streams{ 2 };
const std::vector<bool>   get_vs_set{ true, false };
const std::vector<size_t> num_requests{ 1, 8, 16, 64 };
const std::vector<size_t> num_batch{ 1, 8, 32, 256 };
using namespace AutoBatchingTests;

namespace AutoBatchingTests {

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_GPU, AutoBatching_Test,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                 ::testing::ValuesIn(get_vs_set),
                                 ::testing::ValuesIn(num_streams),
                                 ::testing::ValuesIn(num_requests),
                                 ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_GPU, AutoBatching_Test_DetectionOutput,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                 ::testing::ValuesIn(get_vs_set),
                                 ::testing::ValuesIn(num_streams),
                                 ::testing::ValuesIn(num_requests),
                                 ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test_DetectionOutput::getTestCaseName);
}  // namespace AutoBatchingTests
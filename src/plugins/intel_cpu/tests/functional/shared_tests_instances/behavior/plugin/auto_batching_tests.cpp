// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <behavior/plugin/auto_batching_tests.hpp>

const std::vector<bool>   get_vs_set{ true, false };
const std::vector<size_t> num_streams{ 1, 2 };
const std::vector<size_t> num_requests{ 1, 3, 8, 9, 16, 64 };
const std::vector<size_t> num_batch{ 1, 4, 8, 16, 32, 64, 128, 256 };
using namespace AutoBatchingTests;

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_CPU, AutoBatching_Test,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(get_vs_set),
                ::testing::ValuesIn(num_streams),
                ::testing::ValuesIn(num_requests),
                ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_CPU, AutoBatching_Test_DetectionOutput,
                         ::testing::Combine(
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::ValuesIn(get_vs_set),
                                 ::testing::ValuesIn(num_streams),
                                 ::testing::ValuesIn(num_requests),
                                 ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test_DetectionOutput::getTestCaseName);
}  // namespace

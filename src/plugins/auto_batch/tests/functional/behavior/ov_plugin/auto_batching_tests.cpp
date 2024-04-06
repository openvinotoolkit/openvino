// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/auto_batching_tests.hpp"

#include "behavior/compiled_model/properties.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<bool> get_vs_set{true, false};
const std::vector<size_t> num_streams{1, 2};
const std::vector<size_t> num_requests{1, 3, 8, 9, 16, 64};
const std::vector<size_t> num_batch{1, 4, 8, 16, 32, 64, 128, 256};
INSTANTIATE_TEST_SUITE_P(smoke_TEMPLATE_AutoBatching,
                         AutoBatching_Test,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(num_streams),
                                            ::testing::ValuesIn(num_requests),
                                            ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TEMPLATE_AutoBatching,
                         AutoBatching_Test_DetectionOutput,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(num_streams),
                                            ::testing::ValuesIn(num_requests),
                                            ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test_DetectionOutput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_CPU_AutoBatching,
                         AutoBatching_Test,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(num_streams),
                                            ::testing::ValuesIn(num_requests),
                                            ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_CPU_AutoBatching,
                         AutoBatching_Test_DetectionOutput,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(num_streams),
                                            ::testing::ValuesIn(num_requests),
                                            ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test_DetectionOutput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GPU_AutoBatching,
                         AutoBatching_Test,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(num_streams),
                                            ::testing::ValuesIn(num_requests),
                                            ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GPU_AutoBatching,
                         AutoBatching_Test_DetectionOutput,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(get_vs_set),
                                            ::testing::ValuesIn(num_streams),
                                            ::testing::ValuesIn(num_requests),
                                            ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test_DetectionOutput::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
    {ov::auto_batch_timeout(1000)},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_test,
                         OVClassCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_BATCH) + ":" +
                                                              ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(default_properties)),
                         OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

}  // namespace

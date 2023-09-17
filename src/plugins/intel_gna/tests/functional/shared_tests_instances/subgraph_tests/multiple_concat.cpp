// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <subgraph_tests/multiple_concat.hpp>

#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
std::vector<size_t> input_sizes_aligned = {
    64,
    576,
};

std::vector<size_t> constant_sizes_aligned = {
    64,
    32,
};

std::vector<size_t> input_sizes_unaligned = {26, 99};

std::vector<size_t> constant_sizes_unaligned = {26, 99};

std::vector<std::map<std::string, std::string>> additional_config = {
    {{"GNA_COMPACT_MODE", "NO"}, {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "3276.8"}},
    {{"GNA_COMPACT_MODE", "NO"}, {"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_SCALE_FACTOR_0", "3276.8"}},
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(I_aligned_C_aligned,
                         MultipleConcatTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(input_sizes_aligned),
                                            ::testing::ValuesIn(constant_sizes_aligned),
                                            ::testing::ValuesIn(additional_config)),
                         MultipleConcatTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(I_aligned_C_unaligned,
                         MultipleConcatTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(input_sizes_aligned),
                                            ::testing::ValuesIn(constant_sizes_unaligned),
                                            ::testing::ValuesIn(additional_config)),
                         MultipleConcatTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(I_unaligned_C_aligned,
                         MultipleConcatTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(input_sizes_unaligned),
                                            ::testing::ValuesIn(constant_sizes_aligned),
                                            ::testing::ValuesIn(additional_config)),
                         MultipleConcatTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(I_unaligned_C_unaligned,
                         MultipleConcatTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(input_sizes_unaligned),
                                            ::testing::ValuesIn(constant_sizes_unaligned),
                                            ::testing::ValuesIn(additional_config)),
                         MultipleConcatTest::getTestCaseName);
}  // namespace SubgraphTestsDefinitions

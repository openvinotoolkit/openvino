// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <subgraph_tests/multiple_LSTMCell.hpp>
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
std::vector<size_t> input_sizes = {
    80,
    32,
    64,
    25
};

std::vector<size_t> hidden_sizes = {
    64,
    100,
    24,
    32,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_COMPACT_MODE", "NO"},
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "1638.4"},
};
} // namespace


INSTANTIATE_TEST_CASE_P(MultipleLSTMCellTest, MultipleLSTMCellTest,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(input_sizes),
        ::testing::ValuesIn(hidden_sizes),
        ::testing::Values(additional_config)),
    MultipleLSTMCellTest::getTestCaseName);
} // namespace SubgraphTestsDefinitions

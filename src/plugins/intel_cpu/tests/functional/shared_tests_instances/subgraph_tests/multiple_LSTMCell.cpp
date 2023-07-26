// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <subgraph_tests/multiple_LSTMCell.hpp>
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {

std::vector<ngraph::helpers::MemoryTransformation> transformation {
    ngraph::helpers::MemoryTransformation::NONE,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API
};

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
};
} // namespace


INSTANTIATE_TEST_SUITE_P(MultipleLSTMCellTest, MultipleLSTMCellTest,
    ::testing::Combine(
        ::testing::ValuesIn(transformation),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(input_sizes),
        ::testing::ValuesIn(hidden_sizes),
        ::testing::Values(additional_config)),
    MultipleLSTMCellTest::getTestCaseName);
} // namespace SubgraphTestsDefinitions

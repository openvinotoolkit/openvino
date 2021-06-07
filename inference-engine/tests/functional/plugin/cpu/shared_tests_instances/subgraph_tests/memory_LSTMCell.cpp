// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <subgraph_tests/memory_LSTMCell.hpp>
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
    std::vector<ngraph::helpers::MemoryTransformation> transformation {
            ngraph::helpers::MemoryTransformation::NONE,
            ngraph::helpers::MemoryTransformation::LOW_LATENCY,
            ngraph::helpers::MemoryTransformation::LOW_LATENCY_REGULAR_API,
            ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2,
            ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API
    };

    std::vector<size_t> input_sizes = {
            80,
            32,
            64,
            100,
            25
    };

    std::vector<size_t> hidden_sizes = {
            128,
            200,
            300,
            24,
            32,
    };

    std::map<std::string, std::string> additional_config = {
    };

    INSTANTIATE_TEST_CASE_P(smoke_MemoryLSTMCellTest, MemoryLSTMCellTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(transformation),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::ValuesIn(input_sizes),
                                    ::testing::ValuesIn(hidden_sizes),
                                    ::testing::Values(additional_config)),
                            MemoryLSTMCellTest::getTestCaseName);
} // namespace SubgraphTestsDefinitions

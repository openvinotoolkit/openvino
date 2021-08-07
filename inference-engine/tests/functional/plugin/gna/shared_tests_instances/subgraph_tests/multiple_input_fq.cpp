// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <subgraph_tests/multiple_input_fq.hpp>
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
std::vector<size_t> input = {
    64,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
};
} // namespace

INSTANTIATE_TEST_SUITE_P(smoke_multiple_input, MultipleInputTest,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(input),
        ::testing::Values(additional_config)),
    MultipleInputTest::getTestCaseName);
} // namespace SubgraphTestsDefinitions

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <subgraph_tests/multiple_input_fq.hpp>

#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
std::vector<size_t> input = {
    64,
};

std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                           {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};
}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_multiple_input,
                         MultipleInputTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(input),
                                            ::testing::ValuesIn(configs)),
                         MultipleInputTest::getTestCaseName);
}  // namespace SubgraphTestsDefinitions

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <subgraph_tests/multi_input_scale.hpp>

#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
std::vector<size_t> input = {32, 15, 17, 10};

std::vector<std::map<std::string, std::string>> additional_config = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "2"}, {"GNA_SCALE_FACTOR_1", "2"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "2"}, {"GNA_SCALE_FACTOR_1", "1638.4"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};
}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_multiple_input_scale,
                         MultipleInputScaleTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(input),
                                            ::testing::ValuesIn(additional_config)),
                         MultipleInputScaleTest::getTestCaseName);
}  // namespace SubgraphTestsDefinitions

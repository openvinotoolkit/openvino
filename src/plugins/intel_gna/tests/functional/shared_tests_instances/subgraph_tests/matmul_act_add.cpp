// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/matmul_act_add.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
const std::vector<size_t> input_sizes = {25, 30, 50};

const std::vector<InferenceEngine::Precision> net_precisions = {InferenceEngine::Precision::FP32,
                                                                InferenceEngine::Precision::FP16};

std::vector<std::map<std::string, std::string>> additional_config = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                     {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};
}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_MatMulActAdd,
                         MatMulActAddTest,
                         ::testing::Combine(::testing::ValuesIn(input_sizes),
                                            ::testing::ValuesIn(net_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(additional_config)),
                         MatMulActAddTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions

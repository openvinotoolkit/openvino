// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/activation_concats_eltwise.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;
namespace {
std::vector<size_t> input_sizes = {7, 16, 35, 64};

std::vector<size_t> concat_const_sizes = {7, 16, 35, 64};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

INSTANTIATE_TEST_SUITE_P(smoke_CompareRefs,
                         ActivationConcatsEltwise,
                         ::testing::Combine(::testing::ValuesIn(input_sizes),
                                            ::testing::ValuesIn(concat_const_sizes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ActivationConcatsEltwise::getTestCaseName);

}  // namespace

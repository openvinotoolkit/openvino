// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/relu_split_reshape.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<std::vector<size_t>> inputShape = {{1, 1, 64}, {1, 1, 128}};

std::vector<size_t> splitAxis = {2};

std::vector<size_t> splitNum = {2};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::vector<std::map<std::string, std::string>> additional_config = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                     {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

INSTANTIATE_TEST_SUITE_P(smoke_relu_split_reshape,
                         ReluSplitReshape,
                         ::testing::Combine(::testing::ValuesIn(inputShape),
                                            ::testing::ValuesIn(splitAxis),
                                            ::testing::ValuesIn(splitNum),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(additional_config)),
                         ReluSplitReshape::getTestCaseName);
}  // namespace

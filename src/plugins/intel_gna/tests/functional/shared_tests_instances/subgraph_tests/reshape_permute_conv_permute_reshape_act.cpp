// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/reshape_permute_conv_permute_reshape_act.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

std::vector<std::array<size_t, 4>> input_shapes{
    {1, 1, 166, 2},
    {1, 1, 144, 2},
    {1, 1, 288, 2},
    {1, 1, 144, 4},
};

std::vector<std::array<size_t, 2>> kernel_shapes{
    {1, 7},
    {1, 15},
};

std::vector<size_t> output_channels{
    16,
    8,
    4,
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "2340"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

namespace SubgraphTestsDefinitions {
INSTANTIATE_TEST_SUITE_P(smoke_basic,
                         ConvReshapeAct,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(kernel_shapes),
                                            ::testing::ValuesIn(output_channels),
                                            ::testing::ValuesIn(configs)),
                         ConvReshapeAct::getTestCaseName);
}  // namespace SubgraphTestsDefinitions

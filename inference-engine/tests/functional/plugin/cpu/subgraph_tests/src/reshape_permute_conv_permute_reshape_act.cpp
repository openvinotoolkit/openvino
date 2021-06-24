// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/reshape_permute_conv_permute_reshape_act.hpp"
#include "common_test_utils/test_constants.hpp"

std::vector<std::array<size_t, 4>> input_shapes {
    {1, 1, 166, 2},
    {1, 1, 144, 2},
    {1, 1, 288, 2},
    {1, 1, 144, 4},
};

std::vector<std::array<size_t, 2>> kernel_shapes {
    {1, 7},
    {1, 15},
};

std::vector<size_t> output_channels {
    16,
    8,
    4,
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = { };

namespace SubgraphTestsDefinitions {
    INSTANTIATE_TEST_SUITE_P(smoke_basic, ConvReshapeAct,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::ValuesIn(input_shapes),
            ::testing::ValuesIn(kernel_shapes),
            ::testing::ValuesIn(output_channels),
            ::testing::Values(additional_config)),
        ConvReshapeAct::getTestCaseName);
} // namespace SubgraphTestsDefinitions



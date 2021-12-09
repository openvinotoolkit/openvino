// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/split_conv_concat.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

std::vector<std::vector<size_t>> inputShapes = {
    {1, 32, 1, 130},
    {1, 64, 1, 170},
    {1, 32, 1, 1026}
};

INSTANTIATE_TEST_SUITE_P(smoke_SplitConvConcat, SplitConvConcat,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        SplitConvConcat::getTestCaseName);


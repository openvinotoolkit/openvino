// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/split_concat_memory.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8,
};

const std::vector<InferenceEngine::SizeVector> shapes = {
    {1, 8, 3, 2},
    {3, 8, 3, 2},
    {3, 8, 3},
    {3, 8},
};

INSTANTIATE_TEST_CASE_P(smoke_CPU, SplitConcatMemory,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(1),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        SplitConcatMemory::getTestCaseName);
}  // namespace

// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/memory.h"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::SizeVector> inShapes = {
        {1},
        {3},
        {3, 3, 3},
        {2, 3, 4, 5},
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP32,
};

const std::vector<int64_t> iterationCount {1, 3, 10};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTest, MemoryTest,
        ::testing::Combine(
                ::testing::Values(ngraph::helpers::MemoryTransformation::NONE),
                ::testing::ValuesIn(iterationCount),
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        MemoryTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTestV3, MemoryTestV3,
        ::testing::Combine(
                ::testing::Values(ngraph::helpers::MemoryTransformation::NONE),
                ::testing::ValuesIn(iterationCount),
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        MemoryTest::getTestCaseName);

} // namespace

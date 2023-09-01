// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/memory.h"

using namespace LayerTestsDefinitions;

namespace {

std::vector<ov::helpers::MemoryTransformation> transformation {
        ov::helpers::MemoryTransformation::NONE,
        ov::helpers::MemoryTransformation::LOW_LATENCY_V2,
        ov::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API,
        ov::helpers::MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT,
};

const std::vector<InferenceEngine::SizeVector> inShapes = {
        {3},
        {100, 100},
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<int64_t> iterationCount {
        1,
        3,
        10
};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTest, MemoryTest,
        ::testing::Combine(
                ::testing::ValuesIn(transformation),
                ::testing::ValuesIn(iterationCount),
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(ov::test::utils::DEVICE_CPU, "HETERO:CPU")),
        MemoryTest::getTestCaseName);

} // namespace


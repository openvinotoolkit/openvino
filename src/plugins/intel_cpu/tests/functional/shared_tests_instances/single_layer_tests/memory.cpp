// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/memory.h"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

std::vector<ngraph::helpers::MemoryTransformation> transformation{
    ngraph::helpers::MemoryTransformation::NONE,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT,
};

const std::vector<InferenceEngine::SizeVector> inShapes = {
    {3},
    {100, 100},
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<int64_t> iterationCount{1, 3, 10};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTest,
                         MemoryTest,
                         ::testing::Combine(::testing::ValuesIn(transformation),
                                            ::testing::ValuesIn(iterationCount),
                                            ::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU, "HETERO:CPU")),
                         MemoryTest::getTestCaseName);

}  // namespace


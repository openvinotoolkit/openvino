// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/memory.h"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

class MemoryTestGna : public MemoryTest {
    using Super = MemoryTest;

protected:
    void SetUp() override {
        // 'smoke_MemoryTest' suite does not fit well to GNA quantization.
        // Here input values get multiplied for each iteration present in test.
        // This makes scale computation impossible based on first layer.
        // Manual setting for relative threshold has thus been used.
        threshold = 0.936032;
        Super::SetUp();
    }
};

TEST_P(MemoryTestGna, CompareWithRefs) {
    Run();
}

std::vector<ngraph::helpers::MemoryTransformation> transformation{
    ngraph::helpers::MemoryTransformation::NONE,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API,
    ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT};

const std::vector<InferenceEngine::SizeVector> inShapes = {{1, 1}, {1, 2}, {1, 10}};

const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<int64_t> iterationCount{1, 3, 4, 10};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTest,
                         MemoryTestGna,
                         ::testing::Combine(::testing::ValuesIn(transformation),
                                            ::testing::ValuesIn(iterationCount),
                                            ::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         MemoryTest::getTestCaseName);

}  // namespace

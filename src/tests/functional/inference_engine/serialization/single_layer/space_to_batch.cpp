// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/space_to_batch.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

TEST_P(SpaceToBatchLayerTest, Serialize) {
    Serialize();
}

const std::vector<std::vector<int64_t>> blockShapes4D{{1, 1, 2, 2}};
const std::vector<std::vector<int64_t>> padsBegins4D{{0, 0, 0, 0},
                                                     {0, 0, 0, 2}};
const std::vector<std::vector<int64_t>> padsEnds4D{{0, 0, 0, 0}, {0, 0, 0, 2}};
const std::vector<std::vector<size_t>> dataShapes4D{
    {1, 1, 2, 2}, {1, 3, 2, 2}, {1, 1, 4, 4}, {2, 1, 2, 4}};

const auto SpaceToBatch4D = ::testing::Combine(
    ::testing::ValuesIn(blockShapes4D), ::testing::ValuesIn(padsBegins4D),
    ::testing::ValuesIn(padsEnds4D), ::testing::ValuesIn(dataShapes4D),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_spacetobatch4D_Serialization,
                        SpaceToBatchLayerTest, SpaceToBatch4D,
                        SpaceToBatchLayerTest::getTestCaseName);

}  // namespace

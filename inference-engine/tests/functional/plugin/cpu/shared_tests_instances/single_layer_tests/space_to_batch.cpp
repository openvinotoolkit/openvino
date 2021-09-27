// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/space_to_batch.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::vector<int64_t >> blockShapes4D {
        {1, 1, 2, 2}
};
const std::vector<std::vector<int64_t >> padsBegins4D {
        {0, 0, 0, 0}, {0, 0, 0, 2}
};
const std::vector<std::vector<int64_t >> padsEnds4D {
        {0, 0, 0, 0}, {0, 0, 0, 2}
};
const std::vector<std::vector<size_t >> dataShapes4D {
        {1, 1, 2, 2}, {1, 3, 2, 2}, {1, 1, 4, 4}, {2, 1, 2, 4}
};

const auto SpaceToBatch4D = ::testing::Combine(
        ::testing::ValuesIn(blockShapes4D),
        ::testing::ValuesIn(padsBegins4D),
        ::testing::ValuesIn(padsEnds4D),
        ::testing::ValuesIn(dataShapes4D),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_spacetobatch4D, SpaceToBatchLayerTest, SpaceToBatch4D,
        SpaceToBatchLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t >> blockShapes5D {
        {1, 1, 3, 2, 2}
};
const std::vector<std::vector<int64_t >> padsBegins5D {
        {0, 0, 1, 0, 3}
};
const std::vector<std::vector<int64_t >> padsEnds5D {
        {0, 0, 2, 0, 0}
};
const std::vector<std::vector<size_t >> dataShapes5D {
        {1, 1, 3, 2, 1}
};

const auto SpaceToBatch5D = ::testing::Combine(
        ::testing::ValuesIn(blockShapes5D),
        ::testing::ValuesIn(padsBegins5D),
        ::testing::ValuesIn(padsEnds5D),
        ::testing::ValuesIn(dataShapes5D),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_spacetobatch5D, SpaceToBatchLayerTest, SpaceToBatch5D,
        SpaceToBatchLayerTest::getTestCaseName);

}  // namespace
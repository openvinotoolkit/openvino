// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <vector>

#include "shared_test_classes/single_layer/scatter_ND_update.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(ScatterNDUpdateLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::I64,
};
// map<inputShape map<indicesShape, indicesValue>>
// updateShape is gotten from inputShape and indicesShape
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>>
    sliceSelectInShape{
        {{10, 9, 9, 11},
         {{{4, 1}, {1, 3, 5, 7}},
          {{1, 2}, {4, 6}},
          {{2, 3}, {0, 1, 1, 2, 2, 2}},
          {{1, 4}, {5, 5, 4, 9}}}},
        {{10, 9, 10, 9, 10}, {{{2, 2, 1}, {5, 6, 2, 8}}, {{2, 3}, {0, 4, 6, 5, 7, 1}}}},
    };

const auto ScatterNDUpdateCases = ::testing::Combine(
    ::testing::ValuesIn(ScatterNDUpdateLayerTest::combineShapes(sliceSelectInShape)),
    ::testing::ValuesIn(inputPrecisions),
    ::testing::ValuesIn(idxPrecisions),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterNDUpdateLayerTestSerialization,
    ScatterNDUpdateLayerTest,
    ScatterNDUpdateCases,
    ScatterNDUpdateLayerTest::getTestCaseName);
}  // namespace

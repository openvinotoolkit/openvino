// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/opsets/opset3.hpp>

#include "single_layer_tests/scatter_update.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,
};

// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape {
    {{10, 16, 12, 15}, {{{2, 2, 2}, {0, 1, 2, 3}}, {{2, 4}, {0, 1, 2, 3}}, {{8}, {0, 1, 2, 3}}}},
    {{10, 9, 10, 9, 10}, {{{8}, {0, 1, 2, 3, 4}}, {{4, 2}, {0, 1, 2, 3, 4}}}},
    {{10, 9, 10, 9, 10, 12}, {{{8}, {0, 1, 2, 3, 4, 5}}}},
};
//indices should not be random value
const std::vector<std::vector<int64_t>> idxValue = {
        {0, 2, 4, 6, 1, 3, 5, 7}
};

const auto ScatterUpdateCase = ::testing::Combine(
        ::testing::ValuesIn(ScatterUpdateLayerTest::combineShapes(axesShapeInShape)),
        ::testing::ValuesIn(idxValue),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate, ScatterUpdateLayerTest, ScatterUpdateCase, ScatterUpdateLayerTest::getTestCaseName);

}  // namespace

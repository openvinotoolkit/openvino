// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/opsets/opset3.hpp>

#include "single_layer_tests/scatter_ND_update.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {

// map<inputShape map<indicesShape, indicesValue>>
// updateShape is gotten from inputShape and indicesShape
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> sliceSelectInShape{
    {{4, 3, 2, 3, 2}, {{{2, 2, 1}, {3, 2, 0, 1}}}},
    {{10, 9, 9, 11}, {{{4, 1}, {1, 3, 5, 7}}, {{1, 2}, {4, 6}}, {{2, 3}, {0, 1, 1, 2, 2, 2}}, {{1, 4}, {5, 5, 4, 9}}}},
    {{10, 9, 12, 10, 11}, {{{2, 2, 1}, {5, 6, 2, 8}}, {{2, 3}, {0, 4, 6, 5, 7, 1}}}},
    {{15},                        {{{2, 1}, {1, 3}}}},
    {{15, 14},                    {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}}},
    {{15, 14, 13},                {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}}},
    {{15, 14, 13, 12},            {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}, {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
    {{2, 2, 2}, {2, 3, 1, 8, 7, 5, 6, 5}}}},
    {{15, 14, 13, 12, 16},        {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}, {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
    {{2, 5}, {2, 3, 1, 8, 6, 9, 7, 5, 6, 5}}}},
    {{15, 14, 13, 12, 16, 10},    {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}, {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
    {{1, 2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}}, {{2, 5}, {2, 3, 1, 8, 6,  9, 7, 5, 6, 5}}, {{2, 6}, {2, 3, 1, 8, 6, 5,  9, 7, 5, 6, 5, 7}}}}
};


const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterNDUpdate,
    ScatterNDUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(ScatterNDUpdateLayerTest::combineShapes(sliceSelectInShape)),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterNDUpdateLayerTest::getTestCaseName);
}  // namespace

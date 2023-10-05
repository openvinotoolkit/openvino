// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/opsets/opset3.hpp>

#include "single_layer_tests/scatter_elements_update.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape {
    {{10, 12, 15}, {{{1, 2, 4}, {0, 1, 2}}, {{2, 2, 2}, {-1, -2, -3}}}},
    {{15, 9, 8, 12}, {{{1, 2, 2, 2}, {0, 1, 2, 3}}, {{1, 2, 1, 4}, {-1, -2, -3, -4}}}},
    {{9, 9, 8, 8, 11, 10}, {{{1, 2, 1, 2, 1, 2}, {5, -3}}}},
};

// index value should not be random data
const std::vector<std::vector<size_t>> idxValue = {
        {1, 0, 4, 6, 2, 3, 7, 5}
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
    smoke_ScatterEltsUpdate,
    ScatterElementsUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(ScatterElementsUpdateLayerTest::combineShapes(axesShapeInShape)),
                       ::testing::ValuesIn(idxValue),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdateLayerTest::getTestCaseName);


const std::vector<ov::op::v12::ScatterElementsUpdate::Reduction> reduceModes{
    // Reduction::NONE is omitted intentionally, because v12 with Reduction::NONE is converted to v3,
    // and v3 is already tested by smoke_ScatterEltsUpdate testsuite. It doesn't make sense to test the same code twice.
    // Don't forget to add Reduction::NONE when/if ConvertScatterElementsUpdate12ToScatterElementsUpdate3
    // transformation will be disabled (in common transforamtions pipeline or for GPU only).
    ov::op::v12::ScatterElementsUpdate::Reduction::SUM,
    ov::op::v12::ScatterElementsUpdate::Reduction::PROD,
    ov::op::v12::ScatterElementsUpdate::Reduction::MIN,
    ov::op::v12::ScatterElementsUpdate::Reduction::MAX,
    ov::op::v12::ScatterElementsUpdate::Reduction::MEAN
};

const std::vector<std::vector<int64_t>> idxWithNegativeValues = {
    {1, 0, 4, 6, 2, 3, 7, 5},
    {-1, 0, -4, -6, -2, -3, -7, -5},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterEltsUpdate12,
    ScatterElementsUpdate12LayerTest,
    ::testing::Combine(::testing::ValuesIn(ScatterElementsUpdateLayerTest::combineShapes(axesShapeInShape)),
                       ::testing::ValuesIn(idxWithNegativeValues),
                       ::testing::ValuesIn(reduceModes),
                       ::testing::ValuesIn({true, false}),
                       ::testing::Values(inputPrecisions[0]),
                       ::testing::Values(idxPrecisions[0]),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdate12LayerTest::getTestCaseName);
}  // namespace

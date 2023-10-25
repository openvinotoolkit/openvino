// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/opsets/opset3.hpp>

#include "single_layer_tests/space_to_depth.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I16,
};

const std::vector<SpaceToDepth::SpaceToDepthMode> modes = {
        SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
        SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST};

const std::vector<std::vector<size_t >> inputShapesBS2 = {
        {1, 1, 2, 2}, {1, 1, 4, 4}, {1, 1, 6, 6}, {2, 8, 6, 6}, {2, 4, 10, 8},
        {1, 1, 2, 2, 2}, {1, 1, 4, 4, 4}, {1, 1, 6, 6, 6}, {2, 8, 6, 6, 6}, {2, 4, 10, 8, 12}};

INSTANTIATE_TEST_SUITE_P(SpaceToDepthBS2,
                         SpaceToDepthLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesBS2),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(modes),
                                            ::testing::Values(2),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         SpaceToDepthLayerTest::getTestCaseName);

const std::vector<std::vector<size_t >> inputShapesBS3 = {
        {1, 1, 3, 3}, {1, 1, 6, 6}, {1, 1, 9, 9}, {2, 4, 9, 9}, {2, 3, 15, 12},
        {1, 1, 3, 3, 3}, {1, 1, 6, 6, 6}, {1, 1, 9, 9, 9}, {2, 4, 9, 9, 9}, {2, 3, 15, 12, 18}};

INSTANTIATE_TEST_SUITE_P(SpaceToDepthBS3,
                         SpaceToDepthLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesBS3),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(modes),
                                            ::testing::Values(3),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         SpaceToDepthLayerTest::getTestCaseName);

}  // namespace

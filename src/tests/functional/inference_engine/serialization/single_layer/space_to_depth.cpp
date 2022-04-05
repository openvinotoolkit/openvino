// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/space_to_depth.hpp"

#include <ngraph/opsets/opset3.hpp>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
TEST_P(SpaceToDepthLayerTest, Serialize) {
    Serialize();
}
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I16,
};

const std::vector<SpaceToDepth::SpaceToDepthMode> modes = {
    SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
    SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST};

const std::vector<std::vector<size_t>> inputShapesBS2 = {
    {1, 1, 2, 2},    {1, 1, 4, 4},     {1, 1, 6, 6},    {2, 8, 6, 6},
    {2, 4, 10, 8},   {1, 1, 2, 2, 2},  {1, 1, 4, 4, 4}, {1, 1, 6, 6, 6},
    {2, 8, 6, 6, 6}, {2, 4, 10, 8, 12}};

const auto SpaceToDepthBS2 = ::testing::Combine(
    ::testing::ValuesIn(inputShapesBS2), ::testing::ValuesIn(inputPrecisions),
    ::testing::ValuesIn(modes), ::testing::Values(1, 2),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(
    smoke_SpaceToDepthSerialization, SpaceToDepthLayerTest,
    ::testing::Combine(::testing::ValuesIn(inputShapesBS2),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(modes), ::testing::Values(1, 2),
                       ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    SpaceToDepthLayerTest::getTestCaseName);
}  // namespace

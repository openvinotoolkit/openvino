// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/depth_to_space.hpp"

#include <ngraph/opsets/opset3.hpp>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
TEST_P(DepthToSpaceLayerTest, Serialize) {
    Serialize();
}
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I16,
};

const std::vector<DepthToSpace::DepthToSpaceMode> modes = {
    DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
    DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

const std::vector<std::vector<size_t>> inputShapesBS2 = {
    {1, 4, 1, 1},     {1, 4, 2, 2},    {1, 4, 3, 3},    {2, 32, 3, 3},
    {2, 16, 5, 4},    {1, 8, 1, 1, 1}, {1, 8, 2, 2, 2}, {1, 8, 3, 3, 3},
    {2, 32, 3, 3, 3}, {2, 16, 5, 4, 6}};

INSTANTIATE_TEST_SUITE_P(
    smoke_DepthToSpaceSerialization, DepthToSpaceLayerTest,
    ::testing::Combine(::testing::ValuesIn(inputShapesBS2),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(modes), ::testing::Values(1, 2),
                       ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DepthToSpaceLayerTest::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/opsets/opset3.hpp"
#include "single_op_tests/depth_to_space.hpp"

namespace {
using ov::test::DepthToSpaceLayerTest;
using ov::op::v0::DepthToSpace;

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::u8,
        ov::element::f16,
};

const std::vector<DepthToSpace::DepthToSpaceMode> modes = {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

const std::vector<std::vector<ov::Shape>> inputShapesBS2 = {
        {{1, 4, 1, 1}}, {{1, 4, 2, 2}}, {{1, 4, 3, 3}}, {{2, 32, 3, 3}}, {{2, 16, 5, 4}},
        {{1, 8, 1, 1, 1}}, {{1, 8, 2, 2, 2}}, {{1, 8, 3, 3, 3}}, {{2, 32, 3, 3, 3}}, {{2, 16, 5, 4, 6}}};

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS2,
                         DepthToSpaceLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS2)),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(modes),
                                            ::testing::Values(2),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         DepthToSpaceLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> inputShapesBS3 = {
        {{1, 9, 1, 1}}, {{1, 9, 2, 2}}, {{1, 9, 3, 3}}, {{2, 36, 3, 3}}, {{2, 27, 5, 4}},
        {{1, 27, 1, 1, 1}}, {{1, 27, 2, 2, 2}}, {{1, 27, 3, 3, 3}}, {{2, 108, 3, 3, 3}}, {{2, 54, 5, 4, 6}}};

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS3,
                         DepthToSpaceLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS3)),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(modes),
                                            ::testing::Values(3),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         DepthToSpaceLayerTest::getTestCaseName);

}  // namespace

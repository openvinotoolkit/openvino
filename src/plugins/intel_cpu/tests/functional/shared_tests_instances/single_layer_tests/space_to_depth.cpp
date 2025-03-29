// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/space_to_depth.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::SpaceToDepthLayerTest;

namespace {
const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::u8,
        ov::element::i16,
};

const std::vector<ov::op::v0::SpaceToDepth::SpaceToDepthMode> modes = {
        ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
        ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST
};

const auto input_shapes_BS2 = ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{
        {{1, 1, 2, 2}}, {{1, 1, 4, 4}}, {{1, 1, 6, 6}}, {{2, 8, 6, 6}}, {{2, 4, 10, 8}},
        {{1, 1, 2, 2, 2}}, {{1, 1, 4, 4, 4}}, {{1, 1, 6, 6, 6}}, {{2, 8, 6, 6, 6}}, {{2, 4, 10, 8, 12}}
});

const auto space_to_depth_BS2 = ::testing::Combine(
        ::testing::ValuesIn(input_shapes_BS2),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(modes),
        ::testing::Values(1, 2),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepthBS2, SpaceToDepthLayerTest, space_to_depth_BS2, SpaceToDepthLayerTest::getTestCaseName);

const auto input_shapes_BS3 = ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{
        {{1, 1, 3, 3}}, {{1, 1, 6, 6}}, {{1, 1, 9, 9}}, {{2, 4, 9, 9}}, {{2, 3, 15, 12}},
        {{1, 1, 3, 3, 3}}, {{1, 1, 6, 6, 6}}, {{1, 1, 9, 9, 9}}, {{2, 4, 9, 9, 9}}, {{2, 3, 15, 12, 18}}
});

const auto space_to_depth_BS3 = ::testing::Combine(
        ::testing::ValuesIn(input_shapes_BS3),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(modes),
        ::testing::Values(1, 3),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepthBS3, SpaceToDepthLayerTest, space_to_depth_BS3, SpaceToDepthLayerTest::getTestCaseName);

}  // namespace

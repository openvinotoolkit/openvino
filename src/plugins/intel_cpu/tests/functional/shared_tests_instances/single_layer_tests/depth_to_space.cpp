// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/depth_to_space.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::DepthToSpaceLayerTest;
using ov::op::v0::DepthToSpace;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::u8,
        ov::element::i16,
};

const std::vector<ov::op::v0::DepthToSpace::DepthToSpaceMode> modes = {
        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
};

const std::vector<std::vector<ov::Shape>> input_shapes_bs2_static = {
        {{1, 4, 1, 1}},
        {{1, 4, 2, 2}},
        {{1, 4, 3, 3}},
        {{2, 32, 3, 3}},
        {{2, 16, 5, 4}},
        {{1, 8, 1, 1, 1}},
        {{1, 8, 2, 2, 2}},
        {{1, 8, 3, 3, 3}},
        {{2, 32, 3, 3, 3}},
        {{2, 16, 5, 4, 6}}
};

const auto DepthToSpaceBS2 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_bs2_static)),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(modes),
        ::testing::Values(1, 2),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS2, DepthToSpaceLayerTest, DepthToSpaceBS2, DepthToSpaceLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_bs3_static = {
        {{1, 9, 1, 1}},
        {{1, 9, 2, 2}},
        {{1, 9, 3, 3}},
        {{2, 36, 3, 3}},
        {{2, 27, 5, 4}},
        {{1, 27, 1, 1, 1}},
        {{1, 27, 2, 2, 2}},
        {{1, 27, 3, 3, 3}},
        {{2, 108, 3, 3, 3}},
        {{2, 54, 5, 4, 6}}
};

const auto DepthToSpaceBS3 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_bs3_static)),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(modes),
        ::testing::Values(1, 3),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS3, DepthToSpaceLayerTest, DepthToSpaceBS3, DepthToSpaceLayerTest::getTestCaseName);

}  // namespace

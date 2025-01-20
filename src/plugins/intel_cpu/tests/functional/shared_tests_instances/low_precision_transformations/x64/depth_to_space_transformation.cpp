// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/depth_to_space_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

const std::vector<ov::op::v0::DepthToSpace::DepthToSpaceMode> modes = {
        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
};

const std::vector<ov::PartialShape> inputShapesBS2 = {
        {1, 4, 3, 3}, {2, 16, 5, 4}
};

const auto DepthToSpaceBS2 = ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(inputShapesBS2),
    ::testing::Values(ov::test::utils::DEVICE_CPU),
    ::testing::ValuesIn(modes),
    ::testing::Values(2)
);

INSTANTIATE_TEST_SUITE_P(LPT_BS2, DepthToSpaceTransformation, DepthToSpaceBS2, DepthToSpaceTransformation::getTestCaseName);

const std::vector<ov::PartialShape> inputShapesBS3 = {
        {1, 9, 3, 3}, {2, 27, 5, 4}
 };

const auto DepthToSpaceBS3 = ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(inputShapesBS3),
    ::testing::Values(ov::test::utils::DEVICE_CPU),
    ::testing::ValuesIn(modes),
    ::testing::Values(3)
);

INSTANTIATE_TEST_SUITE_P(smoke_LPT_BS3, DepthToSpaceTransformation, DepthToSpaceBS3, DepthToSpaceTransformation::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/depth_to_space_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ov::op::v0::DepthToSpace::DepthToSpaceMode> modes = {
        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
};

const std::vector<ngraph::PartialShape> inputShapesBS2 = {
        {1, 4, 3, 3}, {2, 16, 5, 4}
};

const auto DepthToSpaceBS2 = ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(inputShapesBS2),
    ::testing::Values(ov::test::utils::DEVICE_GPU),
    ::testing::ValuesIn(modes),
    ::testing::Values(2)
);

INSTANTIATE_TEST_SUITE_P(LPT_BS2, DepthToSpaceTransformation, DepthToSpaceBS2, DepthToSpaceTransformation::getTestCaseName);

const std::vector<ngraph::PartialShape> inputShapesBS3 = {
        {1, 9, 3, 3}, {2, 27, 5, 4}
 };

const auto DepthToSpaceBS3 = ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(inputShapesBS3),
    ::testing::Values(ov::test::utils::DEVICE_GPU),
    ::testing::ValuesIn(modes),
    ::testing::Values(3)
);

INSTANTIATE_TEST_SUITE_P(LPT_BS3, DepthToSpaceTransformation, DepthToSpaceBS3, DepthToSpaceTransformation::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fully_connected_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<MatMulShapes> shapes = {
    {
        ov::PartialShape{ 1, 16 },
        ov::PartialShape{ 16, 8 },
        false,
        false
    },
    {
        ov::PartialShape{ 1, 16 },
        ov::PartialShape{ 8, 16 },
        false,
        true
    },
    {
        ov::PartialShape{ 16, 1 },
        ov::PartialShape{ 16, 8 },
        true,
        false
    },
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FullyConnectedTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues)),
    FullyConnectedTransformation::getTestCaseName);
}  // namespace

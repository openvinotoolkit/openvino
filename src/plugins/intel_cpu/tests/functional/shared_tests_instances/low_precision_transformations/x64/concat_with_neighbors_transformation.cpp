// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/concat_with_neighbors_graph_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<ov::PartialShape> shapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatWithNeighborsGraphTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues)),
    ConcatWithNeighborsGraphTransformation::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/reshape_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32
    // ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<ReshapeTransformationParam> params = {
    // 3D -> 4D
    {
        { 1, 3, 32 },
        { 1, 3, 4, 8 },
        { 256ul, ngraph::Shape{ 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        true
    },
    // 4D -> 3D
    {
        { 1, 3, 16, 16 },
        { 1, 3, 256 },
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        true
    },
    // 4D -> 3D
    {
        { 1, 3, 16, 16 },
        { 0, 3, -1 },
        { 256ul, ngraph::Shape{ 1, 3, 1, 1 }, { 0.f }, { 255.f }, { 0.f, 0.f, 0.f }, { 255.f, 25.5f, 2.55f } },
        true
    },
    // 4D -> 2D
    {
        { 1, 3, 4, 8 },
        { 1, -1 },
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        true
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ReshapeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ReshapeTransformation::getTestCaseName);
}  // namespace

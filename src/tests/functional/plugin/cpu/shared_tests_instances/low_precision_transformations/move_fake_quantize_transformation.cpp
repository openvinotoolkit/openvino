// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/move_fake_quantize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true)
};

const std::vector<LayerTestsDefinitions::MoveFakeQuantizeTransformationParam> params = {
  // without operation
  {
        {},
        {},
        {},
        {},
        {},
        {},
        "",
        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {},
        "Concatenation",
        "U8",
        1,
    },
    // with ReLU operation
    {
        {},
        {},
        {},
        {},
        {},
        {},
        "relu",
        { 256ul, {}, { -12.7f }, { 12.7f }, { -12.7f }, { 12.7f }},
        {},
        {},
        "Concatenation",
        "U8",
        1
    },
    // negative axis
    {
        {},
        {},
        {},
        {},
        {},
        {},
        "",
        {256ul, {},  {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
        {},
        {},
        "Concatenation",
        "FP32",
        0
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MoveFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    MoveFakeQuantizeTransformation::getTestCaseName);
}  // namespace

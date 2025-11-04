// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/convolution_transformation.hpp"
#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<LayerTestsDefinitions::ConvolutionTransformationParam> params = {
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        false,
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        false,
        "Convolution",
        "i8"
    }
};

const std::vector<ov::Shape> shapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(params),
        ::testing::Values(false)),
    ConvolutionTransformation::getTestCaseName);

}  // namespace

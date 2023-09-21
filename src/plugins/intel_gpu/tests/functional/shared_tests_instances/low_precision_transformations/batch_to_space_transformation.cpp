// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/batch_to_space_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<BatchToSpaceTransformationParam> params = {
    {
        { 4, 3, 50, 86 },
        { 1, 1, 2, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 1 },
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        "batch_to_space",
        "u8"
    },
    {
        { 4, 3, 50, 86 },
        { 1, 1, 2, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 1 },
        {
            256ul,
            ngraph::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f/2.f, 255.f/3.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f/2.f, 255.f/3.f },
        },
        "batch_to_space",
        "f32"
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, BatchToSpaceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(params)),
    BatchToSpaceTransformation::getTestCaseName);
}  // namespace





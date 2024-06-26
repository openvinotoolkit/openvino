// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/space_to_batch_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<SpaceToBatchTransformationParam> params = {
    {
        { 1, 3, 100, 171 },
        { 1, 1, 2, 2 }, { 0, 0, 2, 2 }, { 0, 0, 2, 3 },
        { 256ul, ov::Shape{ 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        "SpaceToBatch",
        "u8"
    },
    {
        {1, 3, 100, 171},
        { 1, 1, 2, 2 }, { 0, 0, 2, 2 }, { 0, 0, 2, 3 },
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f/2.f, 255.f/3.f },
            { 0.f, 0.f, 0.f },
            { 255.f, 255.f/2.f, 255.f/3.f },
        },
        "SpaceToBatch",
        "f32"
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, SpaceToBatchTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(params)),
    SpaceToBatchTransformation::getTestCaseName);
}  // namespace

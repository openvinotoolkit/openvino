// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>
#include "low_precision_transformations/slice_transformation.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

const std::vector<LayerTestsDefinitions::SliceTransformationParam> params = {
    {
        {
            256ul,
            ov::Shape{ 1, 1, 1, 1 },
            { 0.f },
            { 25.5f },
            { 0.f },
            { 12.8f }
        },
        { 0 }, // start
        { 2147483647 }, // end
        { 2 }, // step
        { 2 }, // axes
        "u8"
    },
    {
        {
            256ul,
            ov::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f / 1.f, 255.f / 2.f, 255.f / 3.f },
            { 0.f, 0.f, 0.f },
            { 255.f / 1.f, 255.f / 2.f, 255.f / 3.f }
        },
        { 0 }, // start
        { 2147483647 }, // end
        { 2 }, // step
        { 2 }, // axes
        "u8"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, SliceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 24, 24 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    SliceTransformation::getTestCaseName);

}  // namespace

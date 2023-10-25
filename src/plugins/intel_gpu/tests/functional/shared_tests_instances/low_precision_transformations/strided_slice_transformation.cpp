// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>

#include "low_precision_transformations/strided_slice_transformation.hpp"


using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
};

const std::vector<LayerTestsDefinitions::StridedSliceTransformationParam> params = {
    // channel slice, tensor quantization
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 12.8f } },
        { 0, 0, 0, 0 }, // begin
        { 1, 2, 1, 1 }, // end
        { 1, 1, 1, 1 }, // strided
        { 1, 0, 1, 1 }, // beginMask
        { 1, 0, 1, 1 }, // endMask
        {},// newAxisMask
        {},// shrinkAxisMask
        {}// elipsisMask
    },
    // special dimension slice, tensor quantization
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 12.8f } },
        { 0, 0, 0, 0 },
        { 1, 3, 20, 24 },
        { 1, 1, 1, 1 },
        { 1, 1, 0, 1 },
        { 1, 1, 0, 1 },
        {},
        {},
        {}
    },
    // channel slice, per-channel quantization
    {
        {
            256ul,
            ngraph::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f },
        },
        { 0, 0, 0, 0 },
        { 1, 2, 1, 1 },
        { 1, 1, 1, 1 },
        { 1, 0, 1, 1 },
        { 1, 0, 1, 1 },
        {},
        {},
        {}
    },
    // channel slice, per-channel quantization
    {
        {
            256ul,
            ngraph::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f },
        },
        { 0, 0 },
        { 1, 2 },
        { 1, 1 },
        { 1, 0 },
        { 1, 0 },
        {},
        {},
        {}
    },
    // special dimension slice, per-channel quantization
    {
        {
            256ul,
            ngraph::Shape{ 1, 3, 1, 1 },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f },
            { 0.f, 0.f, 0.f },
            { 255.f, 25.5f, 2.55f },
        },
        { 0, 0, 0, 0 },
        { 1, 3, 20, 24 },
        { 1, 1, 1, 1 },
        { 1, 1, 0, 1 },
        { 1, 1, 0, 1 },
        {},
        {},
        {}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, StridedSliceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 24, 24 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    StridedSliceTransformation::getTestCaseName);

}  // namespace

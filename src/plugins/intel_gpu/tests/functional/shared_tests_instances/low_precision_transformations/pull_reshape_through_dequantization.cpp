// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/pull_reshape_through_dequantization_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    // ov::element::f16 // TODO: enable f16 test inference (change ngraph function + fp32 to fp16 replacements)
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
};

const std::vector<PullReshapeThroughDequantizationTestValues> params = {
    {
        ov::element::f32,
        { 256ul, {{ 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        {},
        { std::vector<float>{ 2.f }, ov::element::i8, {9, 16}},
        {
            { ov::element::f32, false },
            {},
            { {0.03f}, ov::element::f32, {/* from parameter */}, false }
        },
        { {3, 3, 16, 1} },
        { {2}, ov::element::f32, {1, 1, 16, 1}, false },
        { {2, 3, 0, 1} },
        { {16, 1, 1, 3, 3} },
        ov::element::f32,
        {},
        "output_original",
        "U8"
    },
    {
        ov::element::f32,
        { 256ul, {{ 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        {},
        { std::vector<float>{ 2.f }, ov::element::i8, {9, 16}},
        {
            { ov::element::f32, false },
            { {127.0f}, ov::element::f32, {/* from parameter */}, false},
            { {0.03f}, ov::element::f32, {/* from parameter */}, false }
        },
        { {3, 3, 16, 1} },
        { {2}, ov::element::f32, {1, 1, 16, 1}, false },
        { {2, 3, 0, 1} },
        { {16, 1, 1, 3, 3} },
        ov::element::f32,
        {},
        "output_original",
        "FP32"
    }
};

const std::vector<ov::PartialShape> inputShapes = {
    { 1, 16, 9, 9 },
    { 4, 16, 9, 9 }
};

const std::vector<ov::Shape> dequantizationOnWeightElementwiseConstantShapes = {
    { ov::Shape({1, 16}) }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, PullReshapeThroughDequantizationTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(dequantizationOnWeightElementwiseConstantShapes),
        ::testing::ValuesIn(params)),
    PullReshapeThroughDequantizationTransformation::getTestCaseName);

}  // namespace

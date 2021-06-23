// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/transpose_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<TransposeTransformationTestValues> testValues = {
    // U8: per-tensor quantization
    {
        ngraph::Shape({ 1, 1000, 1, 1}),
        { 0, 2, 3, 1},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ngraph::element::f32,
        {256, {}, {0.f}, {25.5f}, {12.5f}, {25.5f + 12.5f}}
    },
    // U8: per-channel quantization
    {
        ngraph::Shape({ 1, 3, 1, 1}),
        { 0, 2, 3, 1},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ngraph::element::f32,
        {
            256,
            {1, 3, 1, 1},
            {0.f, 0.f, 0.f},
            {25.5f, 25.5f, 25.5f},
            {0.f, 12.5f, 25.5f},
            {25.5f, 25.5f + 12.5f * 2, 25.5f + 12.5f * 4}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, TransposeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    TransposeTransformation::getTestCaseName);
}  // namespace

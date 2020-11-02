// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "low_precision_transformations/mat_mul_with_constant_transformation.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = { ngraph::element::f32 };

std::vector<MatMulWithConstantTransformationTestValues> testValues = {
    {
        { 1, 32 },
        { 256ul, ngraph::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        { 32, 10 },
        std::vector<float>(32 * 10, 1.f),
        { 256ul, ngraph::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} }
    }
};

INSTANTIATE_TEST_CASE_P(LPT, MatMulWithConstantTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    MatMulWithConstantTransformation::getTestCaseName);
}  // namespace

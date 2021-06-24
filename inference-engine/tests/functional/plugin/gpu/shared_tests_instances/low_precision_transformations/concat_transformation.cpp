// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/concat_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8
    {
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} }
    },
    // I8
    {
        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
    },
    // mixed
    {
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
    },
    // FQ with unexpected quantizationLevels
    {
        { 16ul, ngraph::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        { 16ul, ngraph::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} }
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ngraph::Shape({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    ConcatTransformation::getTestCaseName);
}  // namespace

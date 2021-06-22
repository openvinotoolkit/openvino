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
    // ngraph::element::f16
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
    // mixed: U8 + I8
    {
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
    },
    // mixed: I8 + U8
    {
        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} }
    },
    // FQ with unexpected quantizationLevels
    {
        { 16ul, ngraph::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        { 16ul, ngraph::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} }
    },
};

const std::vector<ngraph::Shape> shapes = {
    ngraph::Shape({ 1, 3, 16, 16 }),
    ngraph::Shape({ 4, 3, 16, 16 })
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    ConcatTransformation::getTestCaseName);
}  // namespace

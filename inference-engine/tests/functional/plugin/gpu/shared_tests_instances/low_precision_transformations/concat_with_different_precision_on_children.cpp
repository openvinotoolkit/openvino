// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/concat_with_different_precision_on_children.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<ConcatWithDifferentChildrenTransformationParam> testValues = {
    // U8
    {
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} }
    },
    // I8
    {
        { 256ul, ngraph::Shape({}), {-128.f}, {127.f}, {-1.28f}, {1.27f} },
        { 256ul, ngraph::Shape({}), {-128.f}, {127.f}, {-1.28f / 2}, {1.27f / 2} }
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
    }
};

const std::vector<bool> multiChannel = { true/*, false*/ };

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatWithDifferentChildrenTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 10, 10 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(multiChannel)),
    ConcatWithDifferentChildrenTransformation::getTestCaseName);
}  // namespace

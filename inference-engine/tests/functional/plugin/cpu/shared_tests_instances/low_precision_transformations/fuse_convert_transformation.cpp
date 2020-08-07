// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fuse_convert_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};


const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versions = {
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

const std::vector<FuseConvertTransformationTestValues> testValues = {
    // DON'T fuse Convert to Subtract
    {
        ngraph::Shape({ 1, 10, 1}),
        { 0, 2, 1},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ngraph::element::f32,
        {256, {}, {0.f}, {25.5f}, {12.5f}, {25.5f + 12.5f}}
    },
    // fuse Convert to Subtract
    {
        ngraph::Shape({ 1, 10, 1}),
        { 0, 2, 1},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ngraph::element::f32,
        {256, {}, {0.f}, {25.5f}, {-12.5f}, {12.5f}}
    },
    // fuse Convert to Multiply
    {
        ngraph::Shape({ 1, 10, 1}),
        { 0, 2, 1},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ngraph::element::f32,
        {256, {}, {0.f}, {25.5f}, {0.f}, {25.5f}}
    }
};

INSTANTIATE_TEST_CASE_P(LPT, FuseConvertTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(versions),
        ::testing::ValuesIn(testValues)),
    FuseConvertTransformation::getTestCaseName);
}  // namespace

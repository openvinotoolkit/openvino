// Copyright (C) 2018-2023 Intel Corporation
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
        {},
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {},
        {},
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {}
    },
    // I8
    {
        {},
        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        {},
        {},
        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        {}
    },
    // mixed
    {
        {},
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {},
        {},
        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        {}
    },
    // FQ with unexpected quantizationLevels
    {
        {},
        { 16ul, ngraph::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        {},
        {},
        { 16ul, ngraph::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        {}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    ConcatTransformation::getTestCaseName);
}  // namespace


namespace concat_transformation_mixed {

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f16
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // mixed dequantization: FP32 & FP16
    {
        {},
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {},
        std::make_shared<ngraph::opset1::Constant>(ov::element::u8, ov::Shape{1, 3, 16, 16}, std::vector<float>(3 * 16 * 16, 1.0)),
        {},
        {
            { ov::element::f16 },
            {},
            {{1.f, 2.f, 3.f}, ov::element::f16, ov::Shape{1, 3, 1, 1}},
        },
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    ConcatTransformation::getTestCaseName);
}  // namespace concat_transformation_mixed

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/concat_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8
    {
        {},
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {},
        {},
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {}
    },
    // I8
    {
        {},
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        {},
        {},
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        {}
    },
    // mixed: U8 + I8
    {
        {},
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {},
        {},
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        {}
    },
    // mixed: I8 + U8
    {
        {},
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        {},
        {},
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {}
    },
    // FQ with unexpected quantizationLevels
    {
        {},
        { 14ul, ov::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        {},
        {},
        { 14ul, ov::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        {}
    },
    // FQ with INT4 quantizationLevels
    {
        {},
        { 16ul, ov::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        {},
        {},
        { 16ul, ov::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        {}
    },
    // FQ with INT4+INT8 quantizationLevels
    {
        {},
        { 16ul, ov::Shape({}), {0.f}, {15.f}, {0.f}, {1.5f} },
        {},
        {},
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {}
    },
};

const std::vector<ov::PartialShape> shapes = {
    ov::Shape({ 1, 3, 16, 16 }),
    ov::Shape({ 4, 3, 16, 16 })
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    ConcatTransformation::getTestCaseName);
}  // namespace

namespace concat_transformation_mixed {

const std::vector<ov::element::Type> precisions = {
    ov::element::f16
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // mixed dequantization: FP32 & FP16
    {
        {},
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        {},
        std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{1, 3, 16, 16}, std::vector<float>(3 * 16 * 16, 1.0)),
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
        ::testing::Values(ov::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    ConcatTransformation::getTestCaseName);
}  // namespace concat_transformation_mixed

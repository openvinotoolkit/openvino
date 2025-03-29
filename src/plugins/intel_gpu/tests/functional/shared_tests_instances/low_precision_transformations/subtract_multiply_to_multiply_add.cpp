// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/subtract_multiply_to_multiply_add_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<SubtractMultiplyToMultiplyAddTransformationTestValues> testValues = {
    // U8: Multiply {} => Multiply (ScaleShift)
    {
        {1, 3, 16, 16},
        ov::element::f32,
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
    },
    // U8: Multiply { 1x3x1x1 } => Multiply + Add (ScaleShift)
    {
        {1, 3, 16, 16},
        ov::element::f32,
        {
            256ul,
            ov::Shape({1, 3, 1, 1}),
            {0.f, 0.f, 0.f},
            {2.55f, 2.55f / 2.f, 2.55f / 3.f},
            {0.f, 0.f, 0.f},
            {2.55f, 2.55f / 2.f, 2.55f / 3.f}
        },
    },
    // U8: Subtract + Multiply { 1x3x1x1 } => Multiply + Add (ScaleShift)
    {
        {1, 3, 16, 16},
        ov::element::f32,
        {
            256ul,
            ov::Shape({1, 3, 1, 1}),
            {2.55f / 2, 2.55f / 4.f, 2.55f / 6.f},
            {2.55f, 2.55f / 2.f, 2.55f / 3.f},
            {2.55f / 2, 2.55f / 4.f, 2.55f / 6.f},
            {2.55f, 2.55f / 2.f, 2.55f / 3.f}
        },
    },
     {
        {1, 3, 16, 16},
        ov::element::f32,
        {
            256ul,
            ov::Shape({1}),
            {2.55f / 2},
            {2.55f},
            {2.55f / 2},
            {2.55f}
        },
     },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, SubtractMultiplyToMultiplyAddTransformation,
    ::testing::Combine(
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    SubtractMultiplyToMultiplyAddTransformation::getTestCaseName);

}  // namespace

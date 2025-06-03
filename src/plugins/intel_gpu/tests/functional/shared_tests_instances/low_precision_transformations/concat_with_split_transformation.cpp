// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/concat_with_split_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ConcatWithSplitTransformationParam> testValues = {
    // U8
    {
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} }
    },
    // I8
    {
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
    },
    // mixed: U8 + I8
    {
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
    },
    // mixed: I8 + U8
    {
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatWithSplitTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 6, 10, 10 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    ConcatWithSplitTransformation::getTestCaseName);
}  // namespace

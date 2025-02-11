// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/concat_with_child_and_output.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

const std::vector<ConcatWithChildAndOutputTransformationParam> testValues = {
    // U8
    {
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} }
    },
    // I8
    {
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
        { 256ul, ov::Shape({}), {-1.28f}, {1.27f}, {-1.28f / 2.f}, {1.27f / 2.f} }
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

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatWithChildAndOutputTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 6, 10, 10 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(trasformationParamValues)),
    ConcatWithChildAndOutputTransformation::getTestCaseName);
}  // namespace

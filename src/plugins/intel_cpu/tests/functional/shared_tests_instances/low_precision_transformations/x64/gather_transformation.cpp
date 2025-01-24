// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/gather_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

const std::vector<int> opset_version = {
    1, 7, 8
};

const std::vector<GatherTransformationTestValues> testValues = {
    // U8: per-tensor quantization
    {
        {3, 3, 4},
        {1},
        {0},
        {0},
        std::int64_t{0},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ov::element::f32,
        {256, {}, {0.f}, {25.5f}, {12.5f}, {25.5f + 12.5f}}
    },
    // U8: per-channel quantization
    {
        {1, 3, 5},
        {1},
        {0},
        {0},
        std::int64_t{0},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ov::element::f32,
        {
            256,
            {1, 3, 1},
            {0.f, 0.f, 0.f},
            {25.5f, 25.5f, 25.5f},
            {0.f, 12.5f, 25.5f},
            {25.5f, 25.5f + 12.5f * 2, 25.5f + 12.5f * 4}
        }
    },
    // U8: per-channel quantization, axis match with dequantization channel, dequantization constant shape is less than gather input shape
    {
        {1, 3, 4},
        {1},
        {0},
        {1},
        std::int64_t{0},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ov::element::f32,
        {
            256,
            {3, 1},
            {0.f, 0.f, 0.f},
            {25.5f, 25.5f, 25.5f},
            {0.f, 12.5f, 25.5f},
            {25.5f, 25.5f + 12.5f * 2, 25.5f + 12.5f * 4}
        }
    },
    // 4D
    {
        {3, 4, 100, 2},
        {2},
        {1, 2},
        {0},
        std::int64_t{0},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        ov::element::f32,
        {256, {}, {0.f}, {25.5f}, {12.5f}, {25.5f + 12.5f}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GatherTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(opset_version)),
    GatherTransformation::getTestCaseName);
}  // namespace

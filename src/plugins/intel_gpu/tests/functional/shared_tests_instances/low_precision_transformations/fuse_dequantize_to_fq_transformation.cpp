// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_dequantize_to_fake_quantize_transformation.hpp"
#include <vector>
#include <gtest/gtest.h>

using namespace LayerTestsDefinitions;

namespace {

// clang-format off
const std::vector<FuseDequantizeToFakeQuantizeTransformationTestValues> testValues = {
    // 1) Multiply
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ov::element::f32,
            { },
            ov::element::f32,
            { {}, {}, { 0.01f } },
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Multiply with different input and output shape
    {
        {128, 3},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ov::element::f32,
            { },
            ov::element::f32,
            { {}, {}, { {0.01f, 0.1f, 1.f}, ov::element::f32, {1, 3} } },
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Multiply by zero
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ov::element::f32,
            { },
            ov::element::f32,
            { {}, {}, { {0.01f, 0.f, 0.01f} } },
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Subtract + Multiply
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ov::element::f32,
            { },
            ov::element::f32,
            { {}, { -128 }, { 0.01f } },
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Convert + Subtract + Multiply
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ov::element::f32,
            { },
            ov::element::u8,
            { {ov::element::f32}, { -128 }, { 0.01f } },
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Convert + Subtract + Multiply 2) Add
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ov::element::f32,
            { {128}, ov::element::f32 },
            ov::element::u8,
            { {ov::element::f32}, { -128 }, { 0.01f } },
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // issue #40611 for FP32
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            { },
            { },
            ov::element::i32,
            { {ov::element::f32}, {}, {} },
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } }
        }
    },
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FuseDequantizeToFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    FuseDequantizeToFakeQuantizeTransformation::getTestCaseName);
}  // namespace

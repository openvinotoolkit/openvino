// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "low_precision_transformations/mat_mul_with_constant_transformation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
    ov::element::f32
};

// transpose_a = false, transpose_b = true
std::vector<MatMulWithConstantTransformationTestValues> testValues = {
    // 3D with different values
    {
        { 2, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::f32, ov::Shape{ 2, 4 } },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-128.f}, {127.f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
        "FullyConnected",
        "u8"
    },
    // 3D with dequantize on weights
    {
        { 2, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::i8, ov::Shape{ 2, 4 } },
        {},
        { ov::element::f32, {}, {0.1f} },
        "FullyConnected",
        "u8"
    },
    // 3D with different values
    {
        { 1, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {-10.5f}, {4.5f}, {-10.5f}, {4.5f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::f32, ov::Shape{ 2, 4 } },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-128.f}, {127.f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
        "FullyConnected",
        "u8"
    },
    // 4D with different values
    {
        { 1, 1, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::f32, ov::Shape{ 2, 4 } },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-128.f}, {127.f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
        "FullyConnected",
        "u8"
    },
    // 4D with Dq on weights
    {
        { 1, 1, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::i8, ov::Shape{ 2, 4 } },
        {},
        { ov::element::f32, {}, {{0.1f, 0.01f}, ov::element::f32, ov::Shape{ 2, 1 }} },
        "FullyConnected",
        "u8"
    },
    // 4D with Dq on weights, with convert on scales
    {
        { 1, 1, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::i8, ov::Shape{ 2, 4 } },
        {},
        {
            ov::element::f32,
            {},
            ov::builder::subgraph::DequantizationOperations::Multiply({0.1f, 0.01f}, ov::element::f32, ov::Shape{ 2, 1 })
                .setConstantPrecision(ov::element::f16)
                .setAddConvert(true)
        },
        "FullyConnected",
        "u8"
    },
    // 3D with the same values
    {
        { 1, 3, 4 },
        { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {255.f}, {0.f}, {25.5f} },
        { std::vector<float>(4 * 4, 2.f), ov::element::f32, ov::Shape{ 4, 4 } },
        { 256ul, {{1}, {1}, {1}, {1}}, {-128.f}, {127.f}, {-128.f}, {127.f} },
        { {}, {}, {} },
        "FullyConnected",
        "u8"
    },
    // 2D with subtract on activations
    {
        { 2, 3 },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-10.f}, {5.f}, {-10.f, -5.f}, {5.f, 5.f} },
        { std::vector<float>{1, 2, 3, 4, 5, 6}, ov::element::f32, ov::Shape{ 2, 3 } },
        { 256ul, {{1}, {1}, {1}, {1}}, {-128.f}, {127.f}, {-12.8f}, {12.7f} },
        { {}, {}, {} },
        "FullyConnected",
        "u8"
    },
    // 2D with subtract on activations & Dq on weights
    {
        { 2, 3 },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-10.f}, {5.f}, {-10.f, -5.f}, {5.f, 5.f} },
        { std::vector<float>{1, 2, 3, 4, 5, 6}, ov::element::i8, ov::Shape{ 2, 3 } },
        {},
        { ov::element::f32, {}, {0.1f} },
        "FullyConnected",
        "u8"
    },
    // 2D with unusual subtract on activations & Dq on weights
    {
        { 2, 3 },
        { 256ul, {{1, 1}, {1, 1}, {1, 1}, {1, 1}}, {-128.f}, {383.f}, {0.5f}, {1.5f} },
        { std::vector<float>{1, 2, 3, 4, 5, 6}, ov::element::i8, ov::Shape{ 2, 3 } },
        {},
        { ov::element::f32, {}, {0.1f} },
        "FullyConnected",
        "u8"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MatMulWithConstantTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    MatMulWithConstantTransformation::getTestCaseName);
}  // namespace

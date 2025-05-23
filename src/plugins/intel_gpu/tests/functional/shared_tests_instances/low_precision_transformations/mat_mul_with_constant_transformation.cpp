// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "low_precision_transformations/mat_mul_with_constant_transformation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    ov::element::f16
};

// transpose_a = false, transpose_b = true
std::vector<MatMulWithConstantTransformationTestValues> testValues = {
    {
        { 2, 3, 4 },
        { 256ul, {{1, 3, 1}, {1, 3, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 2.55f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 2.55f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::f32, ov::Shape{ 2, 4 } },
        { 256ul, {{2, 1}, {2, 1}, {2, 1}, {2, 1}}, {-128.f, -12.8f}, {127.f, 12.7f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
        "FullyConnected",
        "U8"
    },
    {
        { 2, 3, 4 },
        { 256ul, {{1, 3, 1}, {1, 3, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 2.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 2.f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::i8, ov::Shape{ 2, 4 } },
        {},
        { ov::element::f32, {}, {0.1f} },
        "FullyConnected",
        "U8"
    },
    {
        { 1, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {-10.5f}, {4.5f}, {-10.5f}, {4.5f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::f32, ov::Shape{ 2, 4 } },
        { 256ul, {{2, 1}, {2, 1}, {2, 1}, {2, 1}}, {-128.f, -12.8f}, {127.f, 12.7f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
        "FullyConnected",
        "U8"
    },
    {
        { 1, 1, 3, 4 },
        { 256ul, {{1, 3, 1}, {1, 3, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f, 0.f, 0.f}, {25.f, 24.f, 25.f}, {0.f, 0.f, 0.f}, {25.f, 24.f, 25.f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::f32, ov::Shape{ 2, 4 } },
        { 256ul, {{2, 1}, {2, 1}, {2, 1}, {2, 1}}, {-128.f, -12.8f}, {127.f, 12.7f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
        "FullyConnected",
        "U8"
    },
    {
        { 1, 1, 3, 4 },
        { 256ul, {{1, 3, 1}, {1, 3, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f, 0.f, 0.f}, {25.f, 24.f, 25.f}, {0.f, 0.f, 0.f}, {25.f, 24.f, 25.f} },
        { std::vector<float>(4 * 2, 2.f), ov::element::i8, ov::Shape{ 2, 4 } },
        {},
        { ov::element::f32, {}, {{0.1f, 0.01}, ov::element::f32, ov::Shape{ 2, 1 }} },
        "FullyConnected",
        "U8"
    },
    {
        { 1, 3, 4 },
        { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {255.f}, {0.f}, {25.5f} },
        { std::vector<float>(4 * 4, 2.f), ov::element::f32, ov::Shape{ 4, 4 } },
        { 256ul, {{1}, {1}, {1}, {1}}, {-128.f}, {127.f}, {-128.f}, {127.f} },
        { {}, {}, {} },
        "FullyConnected",
        "U8"
    },
    {
        { 2, 3 },
        { 256ul, {{2, 1}, {2, 1}, {2, 1}, {2, 1}}, {-10.f, -5.f}, {5.f, 5.f}, {-10.f, -5.f}, {5.f, 5.f} },
        { std::vector<float>{1, 2, 3, 4, 5, 6}, ov::element::f32, ov::Shape{ 2, 3 } },
        { 256ul, {{1}, {1}, {1}, {1}}, {-128.f}, {127.f}, {-12.8f}, {12.7f} },
        { {}, {}, {} },
        "FullyConnected",
        "U8"
    },
    {
        { 2, 3 },
        { 256ul, {{2, 1}, {2, 1}, {2, 1}, {2, 1}}, {-10.f, -5.f}, {5.f, 5.f}, {-10.f, -5.f}, {5.f, 5.f} },
        { std::vector<float>{1, 2, 3, 4, 5, 6}, ov::element::i8, ov::Shape{ 2, 3 } },
        {},
        { ov::element::f32, {}, {0.1f} },
        "FullyConnected",
        "U8"
    },
    {
        { 2, 3 },
        { 256ul, {{1, 1}, {1, 1}, {1, 1}, {1, 1}}, {-128.f}, {383.f}, {0.5f}, {1.5f} },
        { std::vector<float>{1, 2, 3, 4, 5, 6}, ov::element::i8, ov::Shape{ 2, 3 } },
        {},
        { ov::element::f32, {}, {0.1f} },
        "FullyConnected",
        "U8"
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MatMulWithConstantTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    MatMulWithConstantTransformation::getTestCaseName);
}  // namespace

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/add_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<LayerTestsDefinitions::AddTestValues> params = {
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        {},
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.8f }, { 12.7f } },
        {},
        -1,
        false,
        {ngraph::element::i8}, {ngraph::element::f32, ngraph::element::i8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        {},
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        {},
        -1,
        false,
        {ngraph::element::i8}, {ngraph::element::f32, ngraph::element::i8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        {},
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -128.f }, { 127.f } },
        {},
        -1,
        true,
        {ngraph::element::i8}, {ngraph::element::i8, ngraph::element::f32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -12.8f }, { 12.7f } },
        {},
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        {},
        -1,
        true,
        {ngraph::element::i8}, {ngraph::element::i8, ngraph::element::f32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        {},
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.7f }, { 12.8f } },
        {},
        -1,
        false,
        {ngraph::element::u8}, {ngraph::element::f32, ngraph::element::u8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        {},
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        {},
        -1,
        false,
        {ngraph::element::u8}, {ngraph::element::f32, ngraph::element::u8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        {},
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -127.f }, { 128.f } },
        {},
        -1,
        true,
        {ngraph::element::u8}, {ngraph::element::u8, ngraph::element::f32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -12.8f }, { 12.7f } },
        {},
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        {},
        -1,
        true,
        {ngraph::element::u8}, {ngraph::element::u8, ngraph::element::f32}
    },
    { {}, {}, {}, {}, -1, false },
    { {}, {}, {}, {}, -1, true },

    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        {
            { {1.f}, ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}},
            {255ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f / 10.f }, { 127.f / 10.f }, { -128.f / 10.f }, { 127.f / 10.f }},
            {},
            "convolution"
        },
        {},
        {},
        1,
        true,
        {ngraph::element::u8}, {ngraph::element::u8, ngraph::element::f32},
        {
            // AddTransformation handled dequantization Multiply, Add was fuxed as convolution biases,
            // as result Convolution name was changed to
            { "convolution0", "Convolution", "U8" }
        }
    },

    // denormal value in const weights: denormal value will be quantized to zero and doesn't affect
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        {
            { {1.f, 2.f, 3e-44f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}},
            {255ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f / 10.f }, { 127.f / 10.f }, { -128.f / 10.f }, { 127.f / 10.f }},
            {},
            "convolution"
        },
        {},
        {},
        1,
        true,
        {ngraph::element::u8}, {ngraph::element::u8, ngraph::element::f32},
        {
            // AddTransformation handled dequantization Multiply, Add was fuxed as convolution biases,
            // as result Convolution name was changed to
            { "convolution0", "Convolution", "U8" }
        }
    },

    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        {
            { {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, ngraph::element::i8, ngraph::Shape{3, 3, 1, 1}},
            {},
            {{ngraph::element::f32}, {}, {{1.f, 2.f, 3.f}, ngraph::element::f32, {3, 1, 1, 1}} },
            "convolution"
        },
        {},
        {},
        1,
        true,
        {ngraph::element::u8}, {ngraph::element::u8, ngraph::element::f32},
        {
            // AddTransformation handled dequantization Multiply, Add was fused as convolution biases,
            // as result Convolution name was changed to
            { "convolution0", "Convolution", "U8" }
        }
    },

    // denormal value in dequantization multiply: AddTransformation desn't handle Multiply
    {
        { 256ul, ngraph::Shape{ 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        {
            { {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, ngraph::element::i8, ngraph::Shape{3, 3, 1, 1}},
            {},
            {{ngraph::element::f32}, {}, {{1.f, 2.e-44, 3.f}, ngraph::element::f32, {3, 1, 1, 1}} },
            "convolution"
        },
        {},
        {},
        1,
        true,
        {ngraph::element::u8}, {ngraph::element::u8, ngraph::element::f32},
        {
            // AddTransformation didn't handle dequantization Multiply, Add was not fused as biases,
            // as result Convolution keeps original name
            { "convolution0", "Convolution", "U8" }
        }
    },
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, AddTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(params)),
    AddTransformation::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/convolution_qdq_transformation.hpp"
#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
};

const std::vector<LayerTestsDefinitions::ConvolutionQDqTransformationParam> params = {
    // Actual:
    //
    //                        Constant
    //                         |      Constant Constant Constant Constant
    //                         |      /FP32    /FP32    /FP32    /FP32
    // FakeQuantize           FakeQuantize
    //  |FP32                  |FP32
    //  |                      |
    // Convert    Constant    Convert
    //  |U8         |U8        |I8
    //  |           |          |
    // Convert    Convert     Convert  Constant
    //   \FP32    /FP32        |FP32   /I8
    //    \      /             |      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32      |FP32   /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    //
    // Transformed:
    //
    // Parameter  Constant  Constant
    //   \U8      /U8      /I8
    //    \      /        /
    //    Subtract   Subtract
    //      \FP32    /FP32
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            {ov::element::f32},
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>{ 15.f }, ov::element::f32},
        { 255ul, ov::Shape({ 1, 1, 1, 1 }), { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ov::element::f32 },
        { ov::element::i8, false },
        {
            { ov::element::f32, false },
            { {-128.f}, ov::element::f32, {}, false, 1ul, ov::element::i8, true },
            { {0.2f}, ov::element::f32, {}, false }
        },
        "Convolution",
        ov::element::u8.get_type_name()
    },

    // Actual:
    //
    //                        Constant
    //                         |      Constant Constant Constant Constant
    //                         |      /FP32    /FP32    /FP32    /FP32
    // FakeQuantize           FakeQuantize
    //  |FP32                  |FP32
    //  |                      |
    // Convert    Constant    Convert
    //  |U8         |U8        |U8
    //  |           |          |
    // Convert    Convert     Convert  Constant
    //   \FP32    /FP32        |FP32   /U8
    //    \      /             |      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32      |FP32   /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    //
    // Transformed:
    //
    // Parameter  Constant  Constant
    //   \U8      /U8      /U8
    //    \      /        /
    //    Subtract   Subtract
    //      \FP32    /FP32
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            {ov::element::f32},
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>{ 15.f }, ov::element::f32},
        { 256ul, ov::Shape({ 1, 1, 1, 1 }), { 0.f }, { 25.5f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            { ov::element::f32, false },
            { {0.3f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.2f}, ov::element::f32, {}, false }
        },
        "Convolution",
        ov::element::u8.get_type_name()
    },

    // Actual:
    //
    //                        Constant
    //                         |      Constant Constant Constant Constant
    //                         |      /FP32    /FP32    /FP32    /FP32
    // FakeQuantize           FakeQuantize
    //  |FP32                  |FP32
    //  |                      |
    // Convert    Constant    Convert
    //  |U8         |U8        |I8
    //  |           |          |
    // Convert    Convert     Convert
    //   \FP32    /FP32        |FP32
    //    \      /             |
    //    Subtract  Constant   |      Constant
    //      \FP32   /FP32      |       /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    //
    // Transformed:
    //
    // Parameter  Constant
    //   \U8      /U8
    //    \      /
    //    Subtract   Constant
    //      \FP32    /I8
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            {ov::element::f32},
            {},
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>{ 15.f }, ov::element::f32},
        { 255ul, ov::Shape({ 1, 1, 1, 1 }), { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ov::element::f32 },
        { ov::element::i8, false },
        {
            { ov::element::f32, false },
            {},
            { {0.2f}, ov::element::f32, {}, false }
        },
        "Convolution",
        ov::element::u8.get_type_name()
    },

    // Actual:
    //
    // FQ
    //  |FP32
    //  |
    // Convert    Convert   Constant  Constant
    //  |U8        |U8       |U8       |U8
    //  |          |         |         |
    // Convert    Convert   Convert   Convert
    //   \FP32    /FP32      \FP32    /FP32
    //    \      /            \      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32       \FP32   /FP32
    //       \     /             \     /
    //       Multiply           Multiply
    //         \FP32           /FP32
    //          \             /
    //            Convolution
    //
    // Transformed:
    //
    //  FQ        Constant Constant
    //   \U8      /U8      / I8
    //    \      /        /
    //    Subtract   Subtract
    //      \FP32    /FP32
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            { ov::element::f32, false },
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        {{0.5f}, ov::element::i8},
        {},
        {},
        {
            { ov::element::f32, false },
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.2f}, ov::element::f32, {}, false }
        },
        "Convolution",
        ov::element::f32.get_type_name()
    },

    // Actual:
    //
    // FQ
    //  |FP32
    //  |
    // Convert    Convert
    //  |U8        |U8
    //  |          |
    // Convert    Convert   Constant
    //   \FP32    /FP32      \U8
    //    \      /            |
    //    Subtract  Constant  Convert   Constant
    //      \FP32   /FP32       \FP32   /FP32
    //       \     /             \     /
    //       Multiply           Multiply
    //         \FP32           /FP32
    //          \             /
    //            Convolution
    //
    // Transformed:
    //
    //  FQ        Constant Constant
    //   \U8      /U8      / I8
    //    \      /        /
    //    Subtract   Subtract
    //      \FP32    /FP32
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            { ov::element::f32, false },
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        {{0.5f}, ov::element::i8},
        {},
        {},
        {
            { ov::element::f32, false },
            {},
            { {0.2f}, ov::element::f32, {}, false }
        },
        "Convolution",
        ov::element::u8.get_type_name()
    },
};

const std::vector<ov::PartialShape> shapes = {
    { 1, 3, 4, 4 },
    { 4, 3, 4, 4 }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConvolutionQDqTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ConvolutionQDqTransformation::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/groupconvolution_qdq_transformation.hpp"
#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
// clang-format off

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    // ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
};

const std::vector<LayerTestsDefinitions::GroupConvolutionQDqTransformationParam> params = {
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
    //           \        Reshape
    //            \      /
    //           GroupConvolution
    //
    // Transformed:
    //
    //FakeQuantize Constant  Constant
    //   \U8      /U8        /I8
    //    \      /          /
    //    Subtract        Subtract
    //      \FP32         /FP32
    //       \           /
    //        \        Reshape
    //         \      /
    //       GroupConvolution  Constant
    //           \FP32        /FP32
    //            \          /
    //              Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            {ov::element::f32},
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::f32, {6, 2, 5, 5} },
        { 255ul, ov::Shape({ 1, 1, 1, 1 }), { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ov::element::f32 },
        { ov::element::i8, false },
        {
            { ov::element::f32, false },
            { {-128.f}, ov::element::f32, {}, false, 1ul, ov::element::i8, true },
            { {0.2f}, ov::element::f32, {}, false }
        },
        { {3, 2, 2, 5, 5} },
        "output_original",
        "fp32",
        false,
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
    // Convert    Convert     Convert  Constant
    //   \FP32    /FP32        |FP32   /I8
    //    \      /             |      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32      |FP32   /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           \         Reshape
    //            \       /
    //          GroupConvolution  Constant
    //               \FP32       /FP32
    //                \         /
    //                 Multiply
    //
    // Transformed:
    //
    //FakeQuantize Constant  Constant
    //   \U8      /U8        /I8
    //    \      /          /
    //    Subtract        Subtract
    //      \FP32         /FP32
    //       \           /
    //        \        Reshape
    //         \       /
    //       GroupConvolution  Constant
    //            \FP32      /FP32
    //             \        /
    //              Multiply

    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            {ov::element::f32},
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::f32, {6, 2, 5, 5} },
        { 255ul, ov::Shape({ 1, 1, 1, 1 }), { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ov::element::f32 },
        { ov::element::i8, false },
        {
            { ov::element::f32, false },
            { {-128.f}, ov::element::f32, {}, false, 1ul, ov::element::i8, true },
            { {0.2f}, ov::element::f32, {}, false }
        },
        { {3, 2, 2, 5, 5} },
        "output_original",
        "fp32",
        true,
    },

    // Actual:
    //
    //                        Constant
    //                         |      Constant Constant Constant Constant
    //                         |      /FP32    /FP32    /FP32    /FP32
    // FakeQuantize           FakeQuantize
    //  |FP32                  |FP32
    //  |                      |
    // Convert                Convert
    //  |U8                    |I8
    //  |                      |
    // Convert    Constant    Convert  Constant
    //    \FP32   /FP32        |FP32   /F32
    //     \     /             |      /
    //      Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           \        Reshape
    //            \       /
    //           GroupConvolution
    //
    // Transformed:
    //
    // FakeQuantize Constant
    //     \U8     /I8
    //      \     /
    //   GroupConvolution  Constant
    //          \FP32      /FP32
    //           \        /
    //            Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            {ov::element::f32},
            {},
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::f32, {6, 2, 5, 5} },
        { 255ul, ov::Shape({ 1, 1, 1, 1 }), { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ov::element::f32 },
        { ov::element::i8, false },
        {
            { ov::element::f32, false },
            {},
            { {0.2f}, ov::element::f32, {}, false }
        },
        { {3, 2, 2, 5, 5} },
        "output_original",
        "u8",
        false
    },

    // Actual:
    //
    //                        Constant
    //                         |      Constant Constant Constant Constant
    //                         |      /FP32    /FP32    /FP32    /FP32
    // FakeQuantize           FakeQuantize
    //  |FP32                  |FP32
    //  |                      |
    // Convert                Convert
    //  |U8                    |I8
    //  |                      |
    // Convert    Constant    Convert  Constant
    //    \FP32   /FP32        |FP32   /F32
    //     \     /             |      /
    //      Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           \        Reshape
    //            \       /
    //           GroupConvolution  Constant
    //                  \FP32      /FP32
    //                   \        /
    //                    Multiply
    //
    // Transformed:
    //
    // FakeQuantize Constant
    //     \U8     /I8
    //      \     /
    //   GroupConvolution  Constant
    //          \FP32      /FP32
    //           \        /
    //            Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            {ov::element::f32},
            {},
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::f32, {6, 2, 5, 5}},
        { 255ul, ov::Shape({ 1, 1, 1, 1 }), { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ov::element::f32 },
        { ov::element::i8, false },
        {
            { ov::element::f32, false },
            {},
            { {0.2f}, ov::element::f32, {}, false }
        },
        { {3, 2, 2, 5, 5} },
        "output_original",
        "u8",
        true,
    },

    // Actual:
    //
    // FQ
    //  |FP32
    //  |
    // Convert    Convert   Constant  Constant
    //  |U8        |U8       |I8       |I8
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
    //           \           Reshape
    //            \         /
    //          GroupConvolution
    //
    // Transformed:
    //
    //  FQ        Constant Constant
    //   \U8      /U8      / I8
    //    \      /        /
    //    Subtract     Subtract
    //      \FP32      /FP32
    //       \        /
    //        \     Reshape
    //         \    /
    //       GroupConvolution  Constant
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
        { std::vector<float>(4, 15.f), ov::element::i8, {6, 2, 5, 5} },
        {},
        {},
        {
            { ov::element::f32, false },
            { {127.f}, ov::element::f32, {}, false, 1ul, ov::element::i8, true },
            { {0.2f}, ov::element::f32, {}, false }
        },
        { {3, 2, 2, 5, 5} },
        "output_original",
        "fp32",
        false,
    },

    // Actual:
    //
    // FQ
    //  |FP32
    //  |
    // Convert    Convert   Constant  Constant
    //  |U8        |U8       |I8       |I8
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
    //           \           Reshape
    //            \         /
    //        GroupConvolution  Constant
    //              \FP32       /FP32
    //               \         /
    //                 Multiply
    //
    // Transformed:
    //
    //  FQ        Constant Constant
    //   \U8      /U8      / I8
    //    \      /        /
    //    Subtract     Subtract
    //      \FP32      /FP32
    //       \        /
    //        \     Reshape
    //         \    /
    //   GroupConvolution  Constant
    //         \FP32       /FP32
    //          \         /
    //           Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            { ov::element::f32, false },
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::i8, {6, 2, 5, 5} },
        {},
        {},
        {
            { ov::element::f32, false },
            { {127.f}, ov::element::f32, {}, false, 1ul, ov::element::i8, true },
            { {0.2f}, ov::element::f32, {}, false }
        },
        { {3, 2, 2, 5, 5} },
        "output_original",
        "fp32",
        true,
    },

    // Actual:
    //
    // FQ
    //  |FP32
    //  |
    // Convert    Convert   Constant  Constant
    //  |U8        |U8       |I8       |I8
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
    //           \           /
    //            \         /
    //        GroupConvolution  Constant
    //              \FP32       /FP32
    //               \         /
    //                 Multiply
    //
    // Transformed:
    //
    //  FQ        Constant Constant
    //   \U8      /U8      / I8
    //    \      /        /
    //    Subtract     Subtract
    //      \FP32      /FP32
    //       \        /
    //        \      /
    //         \    /
    //   GroupConvolution  Constant
    //         \FP32       /FP32
    //          \         /
    //           Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            { ov::element::f32, false },
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::i8, {3, 2, 2, 5, 5} },
        {},
        {},
        {
            { ov::element::f32, false },
            { {126.f, 127.f, 126.f, 127.f, 126.f, 127.f}, ov::element::f32, {3, 2, 1, 1, 1}, false, 1ul, ov::element::i8, true },
            { {0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f}, ov::element::f32, {3, 2, 1, 1, 1}, false }
        },
        {},
        "output_original",
        "u8",
        true,
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
    //   '\'FP32    /FP32      '\'U8
    //    '\'      /            '\'
    //    Subtract  Constant  Convert   Constant
    //      '\'FP32   /FP32       '\'FP32   /FP32
    //       '\'     /             '\'     /
    //       Multiply           Multiply
    //         '\'FP32           /FP32
    //          '\'             /
    //           '\'           Reshape
    //            '\'         /
    //          GroupConvolution
    //
    // Transformed:
    //
    //  FQ        Constant
    //   '\'U8      /U8
    //    '\'      /
    //    Subtract
    //      '\'FP32
    //       '\'        Constant
    //        '\'       /I8
    //         '\'     /
    //   GroupConvolution   Constant
    //           '\'FP32      /FP32
    //            '\'        /
    //             Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            { ov::element::f32, false },
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::i8, {6, 2, 5, 5} },
        {},
        {},
        {
            { ov::element::f32, false },
            {},
            { {0.2f}, ov::element::f32, {}, false }
        },
        { {3, 2, 2, 5, 5} },
        "output_original",
        "u8",
        false,
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
    //   '\'FP32    /FP32      '\'U8
    //    '\'      /            '\'
    //    Subtract  Constant  Convert   Constant
    //      '\'FP32   /FP32       '\'FP32   /FP32
    //       '\'     /             '\'     /
    //       Multiply           Multiply
    //         '\'FP32           /FP32
    //          '\'             /
    //           '\'           /
    //            '\'         /
    //          GroupConvolution
    //
    // Transformed:
    //
    //  FQ        Constant
    //   '\'U8      /U8
    //    '\'      /
    //    Subtract
    //      '\'FP32
    //       '\'        Constant
    //        '\'       /I8
    //         '\'     /
    //   GroupConvolution   Constant
    //           '\'FP32      /FP32
    //            '\'        /
    //             Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            { ov::element::f32, false },
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::i8, {3, 2, 2, 5, 5} },
        {},
        {},
        {
            { ov::element::f32, false },
            {},
            { {0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f}, ov::element::f32, {3, 2, 1, 1, 1}, false }
        },
        {},
        "output_original",
        "u8",
        false,
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
    //   '\'FP32    /FP32      '\'I8
    //    '\'      /            '\'
    //    Subtract  Constant  Convert   Constant
    //      '\'FP32   /FP32       '\'FP32   /FP32
    //       '\'     /             '\'     /
    //       Multiply           Multiply
    //         '\'FP32           /FP32
    //          '\'             /
    //           '\'           Reshape
    //            '\'         /
    //      GroupConvolution   Constant
    //             '\'FP32      /FP32
    //              '\'        /
    //               Multiply
    //
    // Transformed:
    //
    //  FQ        Constant
    //   '\'U8      /U8
    //    '\'      /
    //    Subtract
    //       '\'FP32
    //        '\'      Constant
    //         '\'      /I8
    //          '\'    /
    //   GroupConvolution   Constant
    //           '\'FP32      /FP32
    //            '\'        /
    //             Multiply
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { -12.8f }, { 12.7f }, { 0.f }, { 255.f }, ov::element::f32 },
        { ov::element::u8, false },
        {
            { ov::element::f32, false },
            { {128.f}, ov::element::f32, {}, false, 1ul, ov::element::u8, true },
            { {0.1f}, ov::element::f32, {}, false }
        },
        { std::vector<float>(4, 15.f), ov::element::i8, {6, 2, 5, 5} },
        {},
        {},
        {
            { ov::element::f32, false },
            {},
            { {0.2f}, ov::element::f32, {}, false }
        },
        { {3, 2, 2, 5, 5} },
        "output_original",
        "u8",
        true,
    },
};

const std::vector<ov::PartialShape> shapes = {
    { 1, 6, 24, 24 }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GroupConvolutionQDqTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    GroupConvolutionQDqTransformation::getTestCaseName);

// clang-format on
}  // namespace

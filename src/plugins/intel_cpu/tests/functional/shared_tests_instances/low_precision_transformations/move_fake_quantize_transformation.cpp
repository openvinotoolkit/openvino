// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/move_fake_quantize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true)
};

namespace testValues1 {

const std::vector<LayerTestsDefinitions::MoveFakeQuantizeTransformationParam> params = {
    // without operation
    {
        3,
        "",
        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {},
        "Concatenation",
        "U8",
        1,
    },
    // with ReLU operation
    {
        3,
        "relu",
        { 256ul, {}, { -12.7f }, { 12.7f }, { -12.7f }, { 12.7f }},
        {},
        {},
        "Concatenation",
        "U8",
        1
    },
    // Q/DQ
    {
        3,
        "",
        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
        { ngraph::element::u8 },
        {
            { ngraph::element::f32 },
            {},
            { 0.01f }
        },
        "Concatenation",
        "U8",
        1
    },
    // Q/DQ with ReLU
    {
        3,
        "relu",
        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
        { ngraph::element::u8 },
        {
            { ngraph::element::f32 },
            {},
            { 0.01f }
        },
        "Concatenation",
        "U8",
        1
    },
    // multi-chanels
    {
        3,
        "relu",
        {
           256ul,
           {{1, 6, 1, 1}, {1, 6, 1, 1}, {1, 6, 1, 1}, {1, 6, 1, 1}},
           {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
           {2.55f, 2.55f / 2.f, 2.55f / 3.f, 2.55f / 4.f, 2.55f / 5.f, 2.55f / 6.f},
           {-128.f, -128.f, -128.f, -128.f, -128.f, -128.f},
           {127.f, 127.f, 127.f, 127.f, 127.f, 127.f}
        },
        {},
        {},
        "Concatenation",
        "I8",
        1
    },
    // Q/DQ with multi-channels multiply
    {
       3,
       "",
       {
           256ul,
           {{1, 6, 1, 1}, {1, 6, 1, 1}, {1, 6, 1, 1}, {1, 6, 1, 1}},
           {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
           {2.55f, 2.55f / 2.f, 2.55f / 3.f, 2.55f / 4.f, 2.55f / 5.f, 2.55f / 6.f},
           {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
           {255.f, 255.f / 2.f, 255.f / 3.f, 255.f / 4.f, 255.f / 5.f, 255.f / 6.f},
       },
       { ngraph::element::u8 },
       {
           { ngraph::element::f32 },
           {},
           { {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f}, ngraph::element::f32, {1, 6, 1, 1} },
       },
       "Concatenation",
       "U8",
       1
    },
    // Q/DQ with multi-channels subtract
    {
       3,
       "",
       {
           256ul,
           {{1, 6, 1, 1}, {1, 6, 1, 1}, {1, 6, 1, 1}, {1, 6, 1, 1}},
           {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
           {2.55f, 2.55f / 2.f, 2.55f / 3.f, 2.55f / 4.f, 2.55f / 5.f, 2.55f / 6.f},
           {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
           {255.f, 255.f / 2.f, 255.f / 3.f, 255.f / 4.f, 255.f / 5.f, 255.f / 6.f},
       },
       { ngraph::element::u8 },
       {
           { ngraph::element::f32 },
           { {-127.f, -127.f / 2.f, -127.f / 3.f, -127.f / 4.f, -127.f / 5.f, -127.f / 6.f}, ngraph::element::f32, {1, 6, 1, 1} },
           { 0.01f },
       },
       "Concatenation",
       "U8",
       1
    },
};

const std::vector<std::vector<ngraph::PartialShape>> shapes = {
    {{ 1, 1, 16, 16 }, { 1, 2, 16, 16 }, { 1, 3, 16, 16 }},
    {{ 4, 1, 16, 16 }, { 4, 2, 16, 16 }, { 4, 3, 16, 16 }}
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MoveFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn({false, true}),
        ::testing::ValuesIn(params)),
    MoveFakeQuantizeTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {

    const std::vector<LayerTestsDefinitions::MoveFakeQuantizeTransformationParam> params = {
        // negative axis
        {
            3,
            "",
            {256ul, {},  {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
            {},
            {},
            "Concatenation",
            "I8",
            -1
        },
    };
    const std::vector<std::vector<ngraph::PartialShape>> shapes = {
        {{ 1, 1, 16, 16 }, { 1, 1, 16, 16 }, { 1, 1, 16, 16 }},
        {{ 4, 1, 16, 16 }, { 4, 1, 16, 16 }, { 4, 1, 16, 16 }}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_LPT, MoveFakeQuantizeTransformation,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::ValuesIn(shapes),
            ::testing::Values(ov::test::utils::DEVICE_CPU),
            ::testing::ValuesIn(trasformationParamValues),
            ::testing::ValuesIn({false}),
            ::testing::ValuesIn(params)),
        MoveFakeQuantizeTransformation::getTestCaseName);
} // namespace testValues2

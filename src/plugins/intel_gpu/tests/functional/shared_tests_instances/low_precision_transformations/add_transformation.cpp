// Copyright (C) 2018-2023 Intel Corporation
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
    ngraph::element::f16
};

const std::vector<LayerTestsDefinitions::AddTestValues> params = {
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.8f }, { 12.7f } },
        false,
        {ngraph::element::i8}, {ngraph::element::f32, ngraph::element::i8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        {ngraph::element::i8}, {ngraph::element::f32, ngraph::element::i8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -128.f }, { 127.f } },
        true,
        {ngraph::element::i8}, {ngraph::element::i8, ngraph::element::f32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -12.8f }, { 12.7f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        true,
        {ngraph::element::i8}, {ngraph::element::i8, ngraph::element::f32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.7f }, { 12.8f } },
        false,
        {ngraph::element::u8}, {ngraph::element::f32, ngraph::element::u8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        {ngraph::element::u8}, {ngraph::element::f32, ngraph::element::u8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -127.f }, { 128.f } },
        true,
        {ngraph::element::u8}, {ngraph::element::u8, ngraph::element::f32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -12.8f }, { 12.7f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        true,
        {ngraph::element::u8}, {ngraph::element::u8, ngraph::element::f32}
    },
    { {}, {}, false }, { {}, {}, true },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, AddTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(params)),
    AddTransformation::getTestCaseName);
}  // namespace

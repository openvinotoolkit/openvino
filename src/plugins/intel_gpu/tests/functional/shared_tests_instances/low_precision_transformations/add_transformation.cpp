// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/add_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<LayerTestsDefinitions::AddTestValues> params = {
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.8f }, { 12.7f } },
        false,
        {ov::element::i8}, {ov::element::f32, ov::element::i8}
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        {ov::element::i8}, {ov::element::f32, ov::element::i8}
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -128.f }, { 127.f } },
        true,
        {ov::element::i8}, {ov::element::i8, ov::element::f32}
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -12.8f }, { 12.7f } },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        true,
        {ov::element::i8}, {ov::element::i8, ov::element::f32}
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.7f }, { 12.8f } },
        false,
        {ov::element::u8}, {ov::element::f32, ov::element::u8}
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        {ov::element::u8}, {ov::element::f32, ov::element::u8}
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -127.f }, { 128.f } },
        true,
        {ov::element::u8}, {ov::element::u8, ov::element::f32}
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -12.8f }, { 12.7f } },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        true,
        {ov::element::u8}, {ov::element::u8, ov::element::f32}
    },
    { {}, {}, false }, { {}, {}, true },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, AddTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(params)),
    AddTransformation::getTestCaseName);
}  // namespace

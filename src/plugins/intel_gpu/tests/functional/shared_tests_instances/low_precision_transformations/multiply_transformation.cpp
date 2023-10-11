// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/multiply_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<LayerTestsDefinitions::MultiplyTestValues> params = {
    {
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        ngraph::element::undefined, // ngraph::element::i8
        false
    },
    {
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        ngraph::element::undefined, // ngraph::element::u8
        false
    },
    {
        true,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        ngraph::element::undefined, //ngraph::element::u8
        false
    },
    {
        true,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        ngraph::element::undefined, // ngraph::element::i8
        false
    },
    {
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        true,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        ngraph::element::undefined, // ngraph::element::i8
        false
    },
    {
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.28f }, { 1.27f }, { -128.f }, { 1.27f } },
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        ngraph::element::undefined, // ngraph::element::u8
        false
    },
    {
        false,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        true,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -1.27f }, { 1.28f }, { -1.27f }, { 1.28f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        ngraph::element::undefined, // ngraph::element::u8
        false
    },
    { false, {}, false, {}, {}, ngraph::element::undefined /* ngraph::element::f32 */, false },
    { true, {}, true, {}, {}, ngraph::element::undefined /* ngraph::element::f32 */, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MultiplyTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(params)),
    MultiplyTransformation::getTestCaseName);
}  // namespace




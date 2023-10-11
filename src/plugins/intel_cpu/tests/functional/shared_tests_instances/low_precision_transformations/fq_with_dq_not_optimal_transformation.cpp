// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_with_dq_not_optimal_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ov_lpt_models/fake_quantize.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::pass::low_precision;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8AndI8(),
    // LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8AndI8().setUpdatePrecisions(false)
};

const std::vector<FakeQuantizeWithNotOptimalTransformationTestValues> fakeQuantizeOnDataValues = {
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ngraph::element::f32 },
        { ngraph::element::i8, false },
        {
            { ngraph::element::f32, false },
            { {-128.f}, ngraph::element::f32, {}, false, 1ul, ngraph::element::i8, true },
            { {0.1f}, ngraph::element::f32, {}, false }
        },
        {{5.f}, ngraph::element::i8},
        {},
        {},
        {
            { ngraph::element::f32, false },
            { {127.f}, ngraph::element::f32, {}, false, 1ul, ngraph::element::i8, true },
            { {0.3f}, ngraph::element::f32, {}, false }
        },
        {},
        "FP32"
    },
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ngraph::element::f32 },
        { ngraph::element::i8, false },
        {
            { ngraph::element::f32, false },
            {},
            { {0.1f}, ngraph::element::f32, {}, false }
        },
        {{5.f}, ngraph::element::i8},
        {},
        {},
        {
            { ngraph::element::f32, false },
            {},
            { {0.3f}, ngraph::element::f32, {}, false }
        },
        {},
        "I8"
    },
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ngraph::element::f32 },
        { ngraph::element::i8, false },
        {
            { ngraph::element::f32, false },
            { },
            { {0.1f}, ngraph::element::f32, {}, false }
        },
        {{5.f}, ngraph::element::i8},
        {},
        {},
        {
            { ngraph::element::f32, false },
            { {127.f}, ngraph::element::f32, {}, false, 1ul, ngraph::element::i8, true },
            { {0.3f}, ngraph::element::f32, {}, false }
        },
        {},
        "FP32"
    },
    {
        { 256ul, {{ 1, 1, 1, 1 }}, { 0.f }, { 25.5f }, { -128.f }, { 127.f }, ngraph::element::f32 },
        { ngraph::element::i8, false },
        {
            { ngraph::element::f32, false },
            { {-128.f}, ngraph::element::f32, {}, false, 1ul, ngraph::element::i8, true },
            { {0.1f}, ngraph::element::f32, {}, false }
        },
        {{5.f}, ngraph::element::i8},
        {},
        {},
        {
            { ngraph::element::f32, false },
            { },
            { {0.3f}, ngraph::element::f32, {}, false }
        },
        {},
        "U8"
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FakeQuantizeWithNotOptimalTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataValues)),
    FakeQuantizeWithNotOptimalTransformation::getTestCaseName);
}  // namespace

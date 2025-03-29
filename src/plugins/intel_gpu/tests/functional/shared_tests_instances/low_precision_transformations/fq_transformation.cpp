// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ov_lpt_models/fake_quantize.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::pass::low_precision;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<bool> isConvertOnConstants = {
        false,
        true
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    // can not be passed to plugin
    // nGraph: I8 -> FP32 Convert is not supported
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<FakeQuantizeTransformationParam> fakeQuantizeOnDataValues = {
    {
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        "Pooling", "U8"
    },
    {
        { 256ul, { {1ul}, {1ul}, {1ul}, {1ul} }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        "Pooling", "U8"
    },
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { -1.28f }, { 1.27f } },
        "Pooling", "I8"
    },
    // nGraph: dot interval FQ node is being const-folded, so does Pooling
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 2.55f }, { 2.55f } },
        "Pooling", ""
    },
    {
        { 256ul, {}, { -127.5f }, { 0.f }, { -127.5f }, { 0.f } },
        "Pooling", "U8"
    },
    {
        { 16ul, {}, { 0.f }, { 1.5f }, { 0.f }, { 1.5f } },
        "Pooling", "U8"
    },
    {
        { 16ul, {}, { -8.f }, { 7.f }, { -0.8f }, { 0.7f } },
        "Pooling", "I8"
    },
    // nGraph: I8->FP32 Convert is not supported
    // { 256ul, {}, { -1.28f} , { 1.27f }, { -1.28f} , { 1.27f } },
    // { 256ul, { 1ul }, { -1.28f} , { 1.27f } }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 32, 72, 48 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataValues),
        ::testing::ValuesIn(isConvertOnConstants)),
    FakeQuantizeTransformation::getTestCaseName);
}  // namespace

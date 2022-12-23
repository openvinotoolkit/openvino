// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_fake_quantize_transformation.hpp"

#include <vector>
#include <gtest/gtest.h>

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {

const std::vector<FuseFakeQuantizeTransformationTestValues> testValues = {
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ngraph::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ngraph::element::f32 },
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ngraph::element::f32 }
        },
        {
            { "fakeQuantize1" },
            { "fakeQuantize2" }, // was fused to fakeQuantize1
            2ull
        }
    },
    // pipeline test: both Convolutions have to be in U8 independently of possible fuse
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ngraph::element::f32,
            { 256ul, {}, { -1.27f }, { 1.28f }, { -1.27f }, { 1.28f }, ngraph::element::f32 },
            { 256ul, {}, { -1.27f }, { 1.28f }, { -1.27f }, { 1.28f }, ngraph::element::f32 }
        },
        {
            { "fakeQuantize1", "fakeQuantize2" }, // not fused
            { },
            2ull
        }
    },
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ngraph::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ngraph::element::f32 },
            { 256ul, {}, { 0.f }, { 2.55f / 2.f }, { 0.f }, { 2.55f / 2.f }, ngraph::element::f32 }
        },
        {
            { "fakeQuantize1", "fakeQuantize2" }, // not fused
            { },
            2ull
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FuseFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    FuseFakeQuantizeTransformation::getTestCaseName);

}  // namespace

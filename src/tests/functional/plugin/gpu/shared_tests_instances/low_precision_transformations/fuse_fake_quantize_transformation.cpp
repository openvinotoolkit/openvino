// Copyright (C) 2018-2023 Intel Corporation
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
            { "fakeQuantize2" }
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
            { "fakeQuantize1", "fakeQuantize2"},
            { }
        }
    },
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ngraph::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ngraph::element::f32 },
            { 256ul, {}, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f }, ngraph::element::f32 }
        },
        {
            { "fakeQuantize1", "fakeQuantize2"},
            { }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FuseFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    FuseFakeQuantizeTransformation::getTestCaseName);

}  // namespace

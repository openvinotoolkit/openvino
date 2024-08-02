// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/eliminate_fake_quantize_transformation.hpp"

#include <vector>
#include <gtest/gtest.h>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<EliminateFakeQuantizeTransformationTestValues> testValues = {
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 255.f / 2.f }, { 0.f }, { 255.f / 2.f }, ov::element::f32 },
            { 256ul, {}, { 0.f }, { 255.f / 2.f }, { 0.f }, { 255.f / 2.f }, ov::element::f32 }
        },
        {
            { "fakeQuantize1" },
            { "fakeQuantize2" }, // was fused to fakeQuantize1
            2ull
        }
    },
    {
        {1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            ov::element::f32,
            { 256ul, {}, { 0.f }, { 255.f / 2.f }, { 0.f }, { 255.f / 2.f }, ov::element::f32 },
            { 256ul, {}, { 0.f }, { 255.f / 2.1f }, { 0.f }, { 255.f / 2.1f }, ov::element::f32 }
        },
        {
            { "fakeQuantize1", "fakeQuantize2" }, // not fused
            { },
            2ull
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         EliminateFakeQuantizeTransformation,
                         ::testing::Combine(
                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                             ::testing::ValuesIn(testValues)),
                         EliminateFakeQuantizeTransformation::getTestCaseName);

}  // namespace

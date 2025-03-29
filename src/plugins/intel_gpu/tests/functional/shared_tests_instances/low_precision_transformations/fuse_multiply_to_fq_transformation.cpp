// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_multiply_to_fake_quantize_transformation.hpp"
#include <vector>
#include <gtest/gtest.h>

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<FuseMultiplyToFakeQuantizeTransformationTestValues> testValues = {
        {
            {1, 3, 16, 16},
            LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
            {
                { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
                { {}, {}, {} },
            }
        },
        {
            {1, 3, 16, 16},
            LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
            {
                { 256ul, {}, { -1.28f }, { 1.27f }, { 10.f }, { 2.55f } },
                { {}, {}, {} },
            }
        },
    };

    INSTANTIATE_TEST_SUITE_P(smoke_LPT, FuseMultiplyToFakeQuantizeTransformation,
        ::testing::Combine(
            ::testing::Values(ov::test::utils::DEVICE_GPU),
            ::testing::ValuesIn(testValues)),
        FuseMultiplyToFakeQuantizeTransformation::getTestCaseName);
}  // namespace

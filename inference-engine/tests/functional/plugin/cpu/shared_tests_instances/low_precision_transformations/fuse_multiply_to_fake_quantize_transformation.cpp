// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_multiply_to_fake_quantize_transformation.hpp"
#include <vector>
#include <gtest/gtest.h>

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;
using namespace ngraph;

namespace {

const std::vector<FuseMultiplyToFakeQuantizeTransformationTestValues> testValues = {
    {
        Shape{1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            { {}, {}, {} },
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        {
            { 256ul, {}, { -1.28f }, { 1.27f }, { 10.f }, { 2.55f } },
            { {}, {}, {} },
        }
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FuseMultiplyToFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    FuseMultiplyToFakeQuantizeTransformation::getTestCaseName);
}  // namespace

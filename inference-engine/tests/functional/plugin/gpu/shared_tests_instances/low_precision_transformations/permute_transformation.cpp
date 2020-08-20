// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/permute_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<bool> perTensorValues = { true, false };

const std::vector<bool> transposeChannelDimValues = { true, false };

const std::vector<PermuteTransformationTestValues> testValues = {
    // per-tensor: channels are permuted
    {
        LayerTestsUtils::LayerTransformationParamsFactory::createParams(),
        InferenceEngine::SizeVector({ 1, 3, 16, 16 }),
        InferenceEngine::SizeVector({ 0, 2, 1, 3 }),
        {
            { 0.f },
            { 25.5f},
            { 0.f },
            { 25.5f }
        },
        {
            InferenceEngine::Precision::U8,
            true,
            false
        }
    },
    // per-channel: channels are permuted
    {
        LayerTestsUtils::LayerTransformationParamsFactory::createParams(),
        InferenceEngine::SizeVector({ 1, 3, 16, 16 }),
        InferenceEngine::SizeVector({ 0, 2, 1, 3 }),
        {
            { 0.f, 0.f, 0.f },
            { 25.5f, 25.5f / 2.f, 25.5f / 4.f },
            { 0.f, 0.f, 0.f },
            { 25.5f, 25.5f / 2.f, 25.5f / 4.f }
        },
        {
            InferenceEngine::Precision::FP32,
            false,
            false
        }
    },
    // per-channel: channels are not permuted
    {
        LayerTestsUtils::LayerTransformationParamsFactory::createParams(),
        InferenceEngine::SizeVector({ 1, 3, 16, 16 }),
        InferenceEngine::SizeVector({ 0, 1, 3, 2 }),
        {
            { 0.f, 0.f, 0.f },
            { 25.5f, 25.5f / 2.f, 25.5f / 4.f },
            { 0.f, 0.f, 0.f },
            { 25.5f, 25.5f / 2.f, 25.5f / 4.f }
        },
        {
            InferenceEngine::Precision::U8,
            true,
            false
        }
    }
};

INSTANTIATE_TEST_CASE_P(LPT, PermuteTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    PermuteTransformation::getTestCaseName);
}  // namespace

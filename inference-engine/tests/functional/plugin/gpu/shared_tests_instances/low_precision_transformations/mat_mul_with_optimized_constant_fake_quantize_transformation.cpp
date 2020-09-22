// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/mat_mul_with_optimized_constant_fake_quantize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<LayerTestsDefinitions::MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues> params = {
    {
        { 256ul, ngraph::Shape { 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, ngraph::Shape { 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } }
    }
};

const std::vector<std::pair<InferenceEngine::SizeVector, InferenceEngine::SizeVector>> inputShapes = {
    std::pair<InferenceEngine::SizeVector, InferenceEngine::SizeVector>({ InferenceEngine::SizeVector({ 1, 16 }), InferenceEngine::SizeVector({ 10, 16 }) }),
    std::pair<InferenceEngine::SizeVector, InferenceEngine::SizeVector>({ InferenceEngine::SizeVector({ 1, 16 }), InferenceEngine::SizeVector({ 16, 10 }) })
};

INSTANTIATE_TEST_CASE_P(LPT, MatMulWithOptimizedConstantFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(params)),
    MatMulWithOptimizedConstantFakeQuantizeTransformation::getTestCaseName);
}  // namespace

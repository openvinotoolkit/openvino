// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/normalize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

// The test is disabled because CLDNN does not have a Normalize layer implementation in INT8
INSTANTIATE_TEST_CASE_P(DISABLED_LPT, NormalizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 512, 32, 32 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    NormalizeTransformation::getTestCaseName);
}  // namespace

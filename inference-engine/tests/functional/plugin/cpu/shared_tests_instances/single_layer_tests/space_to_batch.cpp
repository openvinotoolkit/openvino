// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/space_to_batch.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

spaceToBatchParamsTuple stb_only_test_cases[] = {
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 2, 2},
                                InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 3, 2, 2},
                                InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 4, 4},
                                InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 2}, {0, 0, 0, 0}, {2, 1, 2, 4},
                                InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32,
                                CommonTestUtils::DEVICE_CPU),
        spaceToBatchParamsTuple({1, 1, 3, 2, 2}, {0, 0, 1, 0, 3}, {0, 0, 2, 0, 0}, {1, 1, 3, 2, 1},
                                InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32,
                                CommonTestUtils::DEVICE_CPU),
};

INSTANTIATE_TEST_CASE_P(
        smoke_MKLDNN, SpaceToBatchLayerTest, ::testing::ValuesIn(stb_only_test_cases),
        SpaceToBatchLayerTest::getTestCaseName);


}  // namespace
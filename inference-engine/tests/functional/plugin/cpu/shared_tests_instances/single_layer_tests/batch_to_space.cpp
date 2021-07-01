// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/batch_to_space.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

batchToSpaceParamsTuple bts_only_test_cases[] = {
        batchToSpaceParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {4, 1, 1, 1},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        batchToSpaceParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {4, 3, 1, 1},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        batchToSpaceParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {4, 1, 2, 2},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        batchToSpaceParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {8, 1, 1, 2},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
        batchToSpaceParamsTuple({1, 1, 3, 2, 2}, {0, 0, 1, 0, 3}, {0, 0, 2, 0, 0}, {12, 1, 2, 1, 2},
                                InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Precision::UNSPECIFIED,
                                InferenceEngine::Layout::ANY,
                                InferenceEngine::Layout::ANY,
                                CommonTestUtils::DEVICE_CPU),
};

INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN, BatchToSpaceLayerTest, ::testing::ValuesIn(bts_only_test_cases),
                        BatchToSpaceLayerTest::getTestCaseName);


}  // namespace
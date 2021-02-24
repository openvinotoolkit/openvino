// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi-device/multi_device_config.hpp"

#include "behavior/preprocessing.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::U16,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32
};

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviourPreprocessingTestsViaSetInput, PreprocessingPrecisionConvertTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::Values(1, 2, 3, 4, 5),   // Number of input tensor channels
                                ::testing::Values(true),            // Use SetInput
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                        PreprocessingPrecisionConvertTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviourPreprocessingTestsViaGetBlob, PreprocessingPrecisionConvertTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::Values(4, 5),       // Number of input tensor channels (blob_copy only supports 4d and 5d tensors)
                                ::testing::Values(false),      // use GetBlob
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                        PreprocessingPrecisionConvertTest::getTestCaseName);
}  // namespace

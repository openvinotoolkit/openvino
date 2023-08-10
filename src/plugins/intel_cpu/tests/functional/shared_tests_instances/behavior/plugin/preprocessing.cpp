// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/preprocessing.hpp"

#ifdef ENABLE_GAPI_PREPROCESSING

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::U16,
    InferenceEngine::Precision::FP32
};

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_SUITE_P(BehaviourPreprocessingTestsViaSetInput, PreprocessingPrecisionConvertTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::Values(1, 2, 3, 4, 5),   // Number of input tensor channels
                                ::testing::Values(true),            // Use SetInput
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                        PreprocessingPrecisionConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(BehaviourPreprocessingTestsViaGetBlob, PreprocessingPrecisionConvertTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::Values(4, 5),       // Number of input tensor channels (blob_copy only supports 4d and 5d tensors)
                                ::testing::Values(false),      // use GetBlob
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                        PreprocessingPrecisionConvertTest::getTestCaseName);
}  // namespace

#endif // ENABLE_GAPI_PREPROCESSING

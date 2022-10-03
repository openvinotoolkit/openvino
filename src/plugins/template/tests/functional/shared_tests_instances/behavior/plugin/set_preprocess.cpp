// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/set_preprocess.hpp"

#ifdef ENABLE_GAPI_PREPROCESSING

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
        {{ InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
           CommonTestUtils::DEVICE_TEMPLATE }}
};

const std::vector<std::map<std::string, std::string>> heteroConfigs = {
        {{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_TEMPLATE }}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(multiConfigs)),
                        InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(heteroConfigs)),
                        InferRequestPreprocessTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> ioPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::U8
};
const std::vector<InferenceEngine::Layout> netLayouts = {
    InferenceEngine::Layout::NCHW,
    // InferenceEngine::Layout::NHWC
};

const std::vector<InferenceEngine::Layout> ioLayouts = {
    InferenceEngine::Layout::NCHW,
    InferenceEngine::Layout::NHWC
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessConversionTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ioPrecisions),
                                ::testing::ValuesIn(ioPrecisions),
                                ::testing::ValuesIn(netLayouts),
                                ::testing::ValuesIn(ioLayouts),
                                ::testing::ValuesIn(ioLayouts),
                                ::testing::Bool(),
                                ::testing::Bool(),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Bool(),
                                ::testing::Bool(),
                                ::testing::ValuesIn(netLayouts),
                                ::testing::Bool(),
                                ::testing::Bool(),
                                ::testing::Values(true), // only SetBlob
                                ::testing::Values(true), // only SetBlob
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

}  // namespace

#endif // ENABLE_GAPI_PREPROCESSING

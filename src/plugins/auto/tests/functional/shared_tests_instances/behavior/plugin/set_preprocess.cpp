// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/set_preprocess.hpp"

#ifdef ENABLE_GAPI_PREPROCESSING

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE}}};

const std::vector<InferenceEngine::Precision> ioPrecisions = {InferenceEngine::Precision::FP32,
                                                              InferenceEngine::Precision::U8};
const std::vector<InferenceEngine::Layout> netLayouts = {
    InferenceEngine::Layout::NCHW,
    // InferenceEngine::Layout::NHWC
};

const std::vector<InferenceEngine::Layout> ioLayouts = {InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         InferRequestPreprocessConversionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(ioPrecisions),
                                            ::testing::ValuesIn(ioPrecisions),
                                            ::testing::ValuesIn(netLayouts),
                                            ::testing::ValuesIn(ioLayouts),
                                            ::testing::ValuesIn(ioLayouts),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         InferRequestPreprocessDynamicallyInSetBlobTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(netLayouts),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(true),  // only SetBlob
                                            ::testing::Values(true),  // only SetBlob
                                            ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         InferRequestPreprocessConversionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(ioPrecisions),
                                            ::testing::ValuesIn(ioPrecisions),
                                            ::testing::ValuesIn(netLayouts),
                                            ::testing::ValuesIn(ioLayouts),
                                            ::testing::ValuesIn(ioLayouts),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         InferRequestPreprocessDynamicallyInSetBlobTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(netLayouts),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(true),  // only SetBlob
                                            ::testing::Values(true),  // only SetBlob
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

}  // namespace

#endif  // ENABLE_GAPI_PREPROCESSING

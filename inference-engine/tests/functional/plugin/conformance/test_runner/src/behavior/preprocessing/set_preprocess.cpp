// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/preprocessing/set_preprocess.hpp"
#include "conformance.hpp"

namespace {

using namespace ConformanceTests;
using namespace BehaviorTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

const std::vector<std::map<std::string, std::string>> heteroConfigs = {
        {{ "TARGET_FALLBACK" , targetDevice}}
};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
        {{ InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , targetDevice}}
};

const std::vector<std::map<std::string, std::string>> autoConfigs = {
        {{ InferenceEngine::KEY_AUTO_DEVICE_LIST , targetDevice}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(configs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(heteroConfigs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(multiConfigs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(autoConfigs)),
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
                            ::testing::Values(targetDevice),
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
                            ::testing::Values(targetDevice),
                            ::testing::ValuesIn(configs)),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessConversionTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(ioPrecisions),
                            ::testing::ValuesIn(ioPrecisions),
                            ::testing::ValuesIn(netLayouts),
                            ::testing::ValuesIn(ioLayouts),
                            ::testing::ValuesIn(ioLayouts),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                            ::testing::ValuesIn(heteroConfigs)),
                    InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::ValuesIn(netLayouts),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                            ::testing::ValuesIn(heteroConfigs)),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPreprocessConversionTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(ioPrecisions),
                            ::testing::ValuesIn(ioPrecisions),
                            ::testing::ValuesIn(netLayouts),
                            ::testing::ValuesIn(ioLayouts),
                            ::testing::ValuesIn(ioLayouts),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                            ::testing::ValuesIn(multiConfigs)),
                    InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::ValuesIn(netLayouts),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                            ::testing::ValuesIn(multiConfigs)),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessConversionTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(ioPrecisions),
                            ::testing::ValuesIn(ioPrecisions),
                            ::testing::ValuesIn(netLayouts),
                            ::testing::ValuesIn(ioLayouts),
                            ::testing::ValuesIn(ioLayouts),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                            ::testing::ValuesIn(autoConfigs)),
                    InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::ValuesIn(netLayouts),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                            ::testing::ValuesIn(autoConfigs)),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

}  // namespace

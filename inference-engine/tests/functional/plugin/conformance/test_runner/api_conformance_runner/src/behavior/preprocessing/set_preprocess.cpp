// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/preprocessing/set_preprocess.hpp"
#include "api_conformance_helpers.hpp"

namespace {

using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ConformanceTests::targetDevice),
                                ::testing::ValuesIn(configs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_HETERO))),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO))),
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
                            ::testing::Values(ConformanceTests::targetDevice),
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
                            ::testing::Values(ConformanceTests::targetDevice),
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
                            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_HETERO))),
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
                            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_HETERO))),
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
                            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
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
                            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

//INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessConversionTest,
//                    ::testing::Combine(
//                            ::testing::ValuesIn(netPrecisions),
//                            ::testing::ValuesIn(ioPrecisions),
//                            ::testing::ValuesIn(ioPrecisions),
//                            ::testing::ValuesIn(netLayouts),
//                            ::testing::ValuesIn(ioLayouts),
//                            ::testing::ValuesIn(ioLayouts),
//                            ::testing::Bool(),
//                            ::testing::Bool(),
//                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
//                            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO)),
//                    InferRequestPreprocessConversionTest::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
//                    ::testing::Combine(
//                            ::testing::ValuesIn(netPrecisions),
//                            ::testing::Bool(),
//                            ::testing::Bool(),
//                            ::testing::ValuesIn(netLayouts),
//                            ::testing::Bool(),
//                            ::testing::Bool(),
//                            ::testing::Values(true), // only SetBlob
//                            ::testing::Values(true), // only SetBlob
//                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
//                            ::testing::ValuesIn(generateConfigs(CommonTestUtils::A))),
//                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

}  // namespace

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <base/behavior_test_utils.hpp>
#include "behavior/set_preprocess.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    using PreprocessBehTest = BehaviorTestsUtils::BehaviorTestsBasic;
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {},
    };

    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{ InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_MYRIAD}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, PreprocessTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            PreprocessTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, PreprocessTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            PreprocessTest::getTestCaseName);

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

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, PreprocessConversionTest,
                                ::testing::Combine(
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::ValuesIn(ioPrecisions),
                                        ::testing::ValuesIn(ioPrecisions),
                                        ::testing::ValuesIn(netLayouts),
                                        ::testing::ValuesIn(ioLayouts),
                                        ::testing::ValuesIn(ioLayouts),
                                        ::testing::Bool(),
                                        ::testing::Bool(),
                                        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                        ::testing::ValuesIn(configs)),
                                PreprocessConversionTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, PreprocessDynamicallyInSetBlobTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Bool(),
                                ::testing::Bool(),
                                ::testing::ValuesIn(netLayouts),
                                ::testing::Bool(),
                                ::testing::Bool(),
                                ::testing::Values(true), // only SetBlob
                                ::testing::Values(true), // only SetBlob
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::ValuesIn(configs)),
                        PreprocessDynamicallyInSetBlobTest::getTestCaseName);


}  // namespace
// Copyright (C) 2018-2023 Intel Corporation
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
            {},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "0"}, {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "1"}}
    };

    const std::vector<std::map<std::string, std::string>> heteroConfigs = {
            {{ "TARGET_FALLBACK" , ov::test::utils::DEVICE_CPU}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                             InferRequestPreprocessTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_HETERO),
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
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
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
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
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
                                ::testing::Values(ov::test::utils::DEVICE_HETERO),
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
                                ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                ::testing::ValuesIn(heteroConfigs)),
                        InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);
}  // namespace

#endif // ENABLE_GAPI_PREPROCESSING

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/io_blob.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "0"}, {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "1"}}
    };

    const std::vector<std::map<std::string, std::string>> Multiconfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_CPU}}
    };

    const std::vector<std::map<std::string, std::string>> Autoconfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_CPU}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                             InferRequestIOBBlobTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestIOBBlobTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(Multiconfigs)),
                             InferRequestIOBBlobTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(Autoconfigs)),
                             InferRequestIOBBlobTest::getTestCaseName);

    std::vector<InferenceEngine::Precision> prcs = {
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP64,
            InferenceEngine::Precision::I4,
            InferenceEngine::Precision::I8,
            InferenceEngine::Precision::I16,
            InferenceEngine::Precision::I32,
            InferenceEngine::Precision::I64,
            InferenceEngine::Precision::U4,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::U16,
            InferenceEngine::Precision::U32,
            InferenceEngine::Precision::U64,
            InferenceEngine::Precision::BF16,
            InferenceEngine::Precision::BIN,
            InferenceEngine::Precision::BOOL,
    };

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobSetPrecisionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::ValuesIn(configs)),
                         InferRequestIOBBlobSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestIOBBlobSetPrecisionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                 ::testing::ValuesIn(Multiconfigs)),
                         InferRequestIOBBlobSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobSetPrecisionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                 ::testing::ValuesIn(Autoconfigs)),
                         InferRequestIOBBlobSetPrecisionTest::getTestCaseName);

    std::vector<InferenceEngine::Layout> layouts = {
            InferenceEngine::Layout::ANY,
            InferenceEngine::Layout::NCHW,
            InferenceEngine::Layout::NHWC,
            InferenceEngine::Layout::NCDHW,
            InferenceEngine::Layout::NDHWC,
            InferenceEngine::Layout::OIHW,
            InferenceEngine::Layout::GOIHW,
            InferenceEngine::Layout::OIDHW,
            InferenceEngine::Layout::GOIDHW,
            InferenceEngine::Layout::SCALAR,
            InferenceEngine::Layout::C,
            InferenceEngine::Layout::CHW,
            InferenceEngine::Layout::HWC,
            InferenceEngine::Layout::HW,
            InferenceEngine::Layout::NC,
            InferenceEngine::Layout::CN,
            InferenceEngine::Layout::BLOCKED,
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobSetLayoutTest,
                             ::testing::Combine(
                                     ::testing::ValuesIn(layouts),
                                     ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                     ::testing::ValuesIn(configs)),
                             InferRequestIOBBlobSetLayoutTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestIOBBlobSetLayoutTest,
                             ::testing::Combine(
                                     ::testing::ValuesIn(layouts),
                                     ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                     ::testing::ValuesIn(Multiconfigs)),
                             InferRequestIOBBlobSetLayoutTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobSetLayoutTest,
                             ::testing::Combine(
                                     ::testing::ValuesIn(layouts),
                                     ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                     ::testing::ValuesIn(Autoconfigs)),
                             InferRequestIOBBlobSetLayoutTest::getTestCaseName);

}  // namespace

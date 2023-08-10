// Copyright (C) 2018-2023 Intel Corporation
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
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , ov::test::utils::DEVICE_CPU}}
    };

    const std::vector<std::map<std::string, std::string>> Autoconfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , ov::test::utils::DEVICE_CPU}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                             InferRequestIOBBlobTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestIOBBlobTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                    ::testing::ValuesIn(Multiconfigs)),
                             InferRequestIOBBlobTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                    ::testing::ValuesIn(Autoconfigs)),
                             InferRequestIOBBlobTest::getTestCaseName);

}  // namespace

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/callback.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {
        {},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "0"}, {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "1"}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCallbackTests,
        ::testing::Combine(
            ::testing::Values(ov::test::utils::DEVICE_CPU),
            ::testing::ValuesIn(configs)),
        InferRequestCallbackTests::getTestCaseName);
}  // namespace

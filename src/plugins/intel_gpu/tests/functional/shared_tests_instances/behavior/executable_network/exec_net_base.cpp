// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_network_base.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_GPU),
                                    ::testing::ValuesIn(configs)),
                            ExecutableNetworkBaseTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> setNetPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16
};

// AUTO:CPU,GPU test case will use cpu plugin which doesn't support FP16
const std::vector<InferenceEngine::Precision> netPrecisionsForAutoCG = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::I32
};

auto configsSetPrc = []() {
    return std::vector<std::map<std::string, std::string>>{
        {},
        {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS,
          InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}}};
};

auto autoBatchConfig = []() {
    return std::vector<std::map<std::string, std::string>>{
        // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_GPU) + "(4)"},
         // no timeout to avoid increasing the test time
         {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "0 "}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecNetSetPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                 ::testing::ValuesIn(configsSetPrc())),
                         ExecNetSetPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, ExecNetSetPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(autoBatchConfig())),
                         ExecNetSetPrecision::getTestCaseName);
}  // namespace

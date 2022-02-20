// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/wait.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {
        {},
        {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}},
};

const std::vector<ov::AnyMap> Multiconfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU)}
};

const std::vector<ov::AnyMap> Autoconfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU)}
};

const std::vector<ov::AnyMap> AutoBatchConfigs = {
        // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(CommonTestUtils::DEVICE_GPU) + "(4)"},
                // no timeout to avoid increasing the test time
                {CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "0 "}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                ::testing::ValuesIn(configs)),
                            OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(Multiconfigs)),
                            OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(Autoconfigs)),
                            OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVInferRequestWaitTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs)),
                            OVInferRequestWaitTests::getTestCaseName);
}  // namespace

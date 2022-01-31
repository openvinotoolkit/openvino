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
        {{ CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU}}
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

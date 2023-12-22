// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/callback.hpp"

using namespace ov::test::behavior;

namespace {
auto configs = []() {
    return std::vector<ov::AnyMap>{
        {},
        {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS,
          InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}},
    };
};

auto autoBatchConfigs = []() {
    return std::vector<ov::AnyMap>{
        // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_GPU) + "(4)"},
         // no timeout to avoid increasing the test time
         {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "0 "}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCallbackTests,
        ::testing::Combine(
            ::testing::Values(ov::test::utils::DEVICE_GPU),
            ::testing::ValuesIn(configs())),
        OVInferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVInferRequestCallbackTests,
         ::testing::Combine(
                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                 ::testing::ValuesIn(autoBatchConfigs())),
         OVInferRequestCallbackTests::getTestCaseName);

}  // namespace

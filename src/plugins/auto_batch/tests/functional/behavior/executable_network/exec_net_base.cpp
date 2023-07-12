// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_network_base.hpp"

using namespace BehaviorTestsDefinitions;
namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};
auto autoBatchConfig = []() {
    return std::vector<std::map<std::string, std::string>>{
        // explicit batch size 4 to avoid fallback to no auto-batching
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
         // no timeout to avoid increasing the test time
         {ov::auto_batch_timeout.name(), "0"}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         ExecNetSetPrecision,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(autoBatchConfig())),
                         ExecNetSetPrecision::getTestCaseName);
}  // namespace

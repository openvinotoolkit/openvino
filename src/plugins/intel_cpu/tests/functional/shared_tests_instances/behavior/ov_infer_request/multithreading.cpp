// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/multithreading.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {
        {},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "0"}, {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "1"}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                            OVInferRequestMultithreadingTests::getTestCaseName);
}  // namespace

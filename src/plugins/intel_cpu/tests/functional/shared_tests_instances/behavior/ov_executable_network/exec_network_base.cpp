// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"
#include "ie_plugin_config.hpp"

using namespace ov::test::behavior;
namespace {

    const std::vector<ov::AnyMap> configs = {
            {},
    };

    const std::vector<ov::AnyMap> heteroConfigs = {
            {ov::device::priorities(ov::test::utils::DEVICE_CPU)}};

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            OVCompiledModelBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVCompiledModelBaseTest,
                             ::testing::Combine(
                                     ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                     ::testing::ValuesIn(heteroConfigs)),
                             OVCompiledModelBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelBaseTestOptional,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            OVCompiledModelBaseTestOptional::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVCompiledModelBaseTestOptional,
                             ::testing::Combine(
                                     ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                     ::testing::ValuesIn(heteroConfigs)),
                             OVCompiledModelBaseTestOptional::getTestCaseName);

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::I16,
            InferenceEngine::Precision::U16
    };

    const std::vector<ov::AnyMap> configSetPrc = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
    };
}  // namespace

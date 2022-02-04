// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/exec_network_base.hpp"
#include "ie_plugin_config.hpp"

using namespace ov::test::behavior;
namespace {

    const std::vector<ov::AnyMap> configs = {
            {},
    };
    const std::vector<ov::AnyMap> multiConfigs = {
            {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}
    };

    const std::vector<ov::AnyMap> heteroConfigs = {
            {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            OVExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            OVExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(multiConfigs)),
                            OVExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVExecutableNetworkBaseTest,
                             ::testing::Combine(
                                     ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                     ::testing::ValuesIn(heteroConfigs)),
                             OVExecutableNetworkBaseTest::getTestCaseName);

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

    const std::vector<ov::AnyMap> AutoConfigsSetPrc = {
            {ov::device::priorities(CommonTestUtils::DEVICE_CPU)},
    };

    const std::vector<ov::AnyMap> MultiConfigsSetPrc = {
            {ov::device::priorities(CommonTestUtils::DEVICE_CPU)},
            {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
    };
}  // namespace

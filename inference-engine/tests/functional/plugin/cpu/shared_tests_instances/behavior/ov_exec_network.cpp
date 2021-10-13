// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_exec_network.hpp"

#include "ie_plugin_config.hpp"

using namespace ov::test;
namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64,
    ov::element::u8,
    ov::element::u16,
    ov::element::u32,
    ov::element::u64,
    ov::element::f16,
    ov::element::f32,
};

const std::vector<std::map<std::string, std::string>> configs = {
    {},
};
const std::vector<std::map<std::string, std::string>> multiConfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_CPU}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVExecNetwork,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                            ::testing::ValuesIn(configs)),
                         OVExecNetwork::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVExecNetwork,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVExecNetwork::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVExecNetwork,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVExecNetwork::getTestCaseName);
}  // namespace

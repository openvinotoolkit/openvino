// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_exec_graph_info.hpp"

#include "ie_plugin_config.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace ov::test;
namespace {
const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16};

const std::vector<std::map<std::string, std::string>> configs = {
    {},
};
const std::vector<std::map<std::string, std::string>> multiConfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_CPU}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVExecGraphTests,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                            ::testing::ValuesIn(configs)),
                         OVExecGraphTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVExecGraphTests,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVExecGraphTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVExecGraphTests,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVExecGraphTests::getTestCaseName);
}  // namespace

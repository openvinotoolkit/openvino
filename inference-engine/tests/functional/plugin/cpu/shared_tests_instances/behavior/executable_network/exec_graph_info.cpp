// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_graph_info.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {},
    };
    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{ InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(multiConfigs)),
                            ExecutableNetworkBaseTest::getTestCaseName);
}  // namespace

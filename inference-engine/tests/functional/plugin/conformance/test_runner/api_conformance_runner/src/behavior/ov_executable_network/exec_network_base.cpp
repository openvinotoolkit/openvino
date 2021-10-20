// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/exec_network_base.hpp"
#include "ie_plugin_config.hpp"
#include "conformance.hpp"

using namespace ov::test::behavior;
namespace {

    const std::vector<std::map<std::string, std::string>> configs = {
            {},
    };

    const std::vector<std::map<std::string, std::string>> generateMultiConfigsExecNetBase() {
        return {{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ConformanceTests::targetDevice}}};
    }

    const std::vector<std::map<std::string, std::string>> generateHeteroConfigsExecNetBase() {
        return {{{"TARGET_FALLBACK", ConformanceTests::targetDevice}}};
    }

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            OVExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(generateMultiConfigsExecNetBase())),
                            OVExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(generateMultiConfigsExecNetBase())),
                            OVExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVExecutableNetworkBaseTest,
                             ::testing::Combine(
                                     ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                     ::testing::ValuesIn(generateHeteroConfigsExecNetBase())),
                             OVExecutableNetworkBaseTest::getTestCaseName);
}  // namespace

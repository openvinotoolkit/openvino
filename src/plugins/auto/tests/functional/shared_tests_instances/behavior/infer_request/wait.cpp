// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/wait.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> Autoconfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_CPU}},
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_GPU}},
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), std::string(ov::test::utils::DEVICE_CPU) + "," + ov::test::utils::DEVICE_GPU}},
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), std::string(ov::test::utils::DEVICE_GPU) + "," + ov::test::utils::DEVICE_CPU}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestWaitTests,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                    ::testing::ValuesIn(Autoconfigs)),
                             InferRequestWaitTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestWaitTests,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                    ::testing::ValuesIn(Autoconfigs)),
                             InferRequestWaitTests::getTestCaseName);

}  // namespace

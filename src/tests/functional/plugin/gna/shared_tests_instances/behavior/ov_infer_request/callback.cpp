// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/callback.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
        {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_GNA}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCallbackTests,
        ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs)),
        OVInferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestCallbackTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(multiConfigs)),
        OVInferRequestCallbackTests::getTestCaseName);
}  // namespace

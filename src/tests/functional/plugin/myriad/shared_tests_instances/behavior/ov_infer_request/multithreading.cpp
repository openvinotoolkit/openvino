// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/multithreading.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

const std::vector<std::map<std::string, std::string>> Multiconfigs = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_MYRIAD}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::ValuesIn(configs)),
                            OVInferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(Multiconfigs)),
                            OVInferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(Multiconfigs)),
                            OVInferRequestMultithreadingTests::getTestCaseName);

}  // namespace

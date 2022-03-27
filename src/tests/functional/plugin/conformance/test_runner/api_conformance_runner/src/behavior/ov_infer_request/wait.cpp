// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/wait.hpp"

#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::conformance::targetDevice),
                                ::testing::ValuesIn(empty_ov_config)),
                            OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(generate_ov_configs(CommonTestUtils::DEVICE_MULTI))),
                            OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(generate_ov_configs(CommonTestUtils::DEVICE_AUTO))),
                            OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVInferRequestWaitTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                 ::testing::ValuesIn(generate_ov_configs(CommonTestUtils::DEVICE_HETERO))),
                         OVInferRequestWaitTests::getTestCaseName);
}  // namespace

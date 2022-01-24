// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/callback.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace BehaviorTestsDefinitions;
using namespace ConformanceTests;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(targetDevice),
                                 ::testing::ValuesIn(emptyConfig)),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                 ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                 ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                 ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_HETERO))),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Batch_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_BATCH))),
                         InferRequestCallbackTests::getTestCaseName);
}  // namespace

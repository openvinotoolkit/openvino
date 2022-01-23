// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/wait.hpp"

using namespace ov::test::behavior;
namespace {
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestWaitTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                ::testing::Values(ov::AnyMap({}))),
        OVInferRequestWaitTests::getTestCaseName);


}  // namespace

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestOnImportedNetwork,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE, "HETERO:TEMPLATE"),
        OVHoldersTestOnImportedNetwork::getTestCaseName);

}  // namespace

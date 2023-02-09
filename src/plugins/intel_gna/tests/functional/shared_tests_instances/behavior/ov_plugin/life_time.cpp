// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVLifeTimeTest,
                         ::testing::Values(CommonTestUtils::DEVICE_GNA),
                         OVLifeTimeTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVLifeTimeTestOnImportedNetwork,
                         ::testing::Values(CommonTestUtils::DEVICE_GNA),
                         OVLifeTimeTestOnImportedNetwork::getTestCaseName);

}  // namespace

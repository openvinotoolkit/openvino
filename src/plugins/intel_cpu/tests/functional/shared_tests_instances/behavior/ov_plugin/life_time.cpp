// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;
namespace {
    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVLifeTimeTest,
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            OVLifeTimeTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests, OVLifeTimeTest,
            ::testing::Values("AUTO:CPU",
                                "MULTI:CPU",
                                //CommonTestUtils::DEVICE_BATCH,
                                "HETERO:CPU"),
            OVLifeTimeTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVLifeTimeTestOnImportedNetwork,
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            OVLifeTimeTestOnImportedNetwork::getTestCaseName);

}  // namespace

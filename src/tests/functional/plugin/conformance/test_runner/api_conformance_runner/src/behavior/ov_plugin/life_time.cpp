// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTest,
        ::testing::Values(ov::test::conformance::targetDevice),
        OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVHoldersTest,
                         ::testing::Values(ov::test::conformance::generate_complex_device_name(CommonTestUtils::DEVICE_HETERO)),
                         OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVHoldersTest,
                         ::testing::Values(ov::test::conformance::generate_complex_device_name(CommonTestUtils::DEVICE_AUTO)),
                         OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVHoldersTest,
                         ::testing::Values(ov::test::conformance::generate_complex_device_name(CommonTestUtils::DEVICE_BATCH)),
                         OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVHoldersTest,
                         ::testing::Values(ov::test::conformance::generate_complex_device_name(CommonTestUtils::DEVICE_MULTI)),
                         OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestOnImportedNetwork,
        ::testing::Values(ov::test::conformance::targetDevice),
        OVHoldersTestOnImportedNetwork::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::conformance::generate_complex_device_name(CommonTestUtils::DEVICE_HETERO)),
                         OVHoldersTestOnImportedNetwork::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::conformance::generate_complex_device_name(CommonTestUtils::DEVICE_AUTO)),
                         OVHoldersTestOnImportedNetwork::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::conformance::generate_complex_device_name(CommonTestUtils::DEVICE_BATCH)),
                         OVHoldersTestOnImportedNetwork::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::conformance::generate_complex_device_name(CommonTestUtils::DEVICE_MULTI)),
                         OVHoldersTestOnImportedNetwork::getTestCaseName);
}  // namespace

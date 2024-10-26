// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"
#include "ov_api_conformance_helpers.hpp"
#include "base/ov_behavior_test_utils.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVHoldersTest,
        ::testing::Values(ov::test::utils::target_device),
        OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin,
                         OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::utils::target_device),
                         MARK_MANDATORY_API_FOR_HW_DEVICE_WITH_PARAM(OVHoldersTestOnImportedNetwork::getTestCaseName));
}  // namespace

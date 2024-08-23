// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_api_conformance_helpers.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "base/ov_behavior_test_utils.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

static std::string (*null_GetTestName)(const testing::TestParamInfo<std::string>& info) = nullptr;

namespace {
//
// OV Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory,
                         OVClassModelTestP,
                         ::testing::Values(ov::test::utils::target_device),
                         MARK_MANDATORY_FOR_QUERY_MODEL(null_GetTestName));

INSTANTIATE_TEST_SUITE_P(ov_plugin,
                         OVClassModelOptionalTestP,
                         ::testing::Values(ov::test::utils::target_device),
                         MARK_MANDATORY_FOR_QUERY_MODEL(null_GetTestName));

// OV Class Query network


INSTANTIATE_TEST_SUITE_P(ov_plugin,
                         OVClassQueryModelTest,
                         ::testing::Values(ov::test::utils::target_device),
                         MARK_MANDATORY_FOR_QUERY_MODEL(null_GetTestName));

}  // namespace

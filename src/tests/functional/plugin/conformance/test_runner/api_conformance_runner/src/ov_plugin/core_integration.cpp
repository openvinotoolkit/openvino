// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory,
                         OVClassModelTestP,
                         ::testing::Values(targetDevice));

INSTANTIATE_TEST_SUITE_P(ov_plugin,
                         OVClassModelOptionalTestP,
                         ::testing::Values(targetDevice));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory,
                         OVClassQueryModelTest,
                         ::testing::Values(targetDevice));
}  // namespace

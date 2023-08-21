// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"

using namespace ov::test::behavior;

namespace {
// IE Class Common tests with <pluginName, target_device params>
//
INSTANTIATE_TEST_SUITE_P(nightly_OVClassModelTestP, OVClassModelTestP, ::testing::Values("GPU"));
INSTANTIATE_TEST_SUITE_P(nightly_OVClassModelOptionalTestP, OVClassModelOptionalTestP, ::testing::Values("GPU"));

// Several devices case
INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestCompileModel,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestQueryModel,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestDefaultCore,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));

// Set config for all GPU devices

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSetGlobalConfigTest, OVClassSetGlobalConfigTest, ::testing::Values("GPU"));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest, OVClassQueryModelTest, ::testing::Values("GPU"));

}  // namespace

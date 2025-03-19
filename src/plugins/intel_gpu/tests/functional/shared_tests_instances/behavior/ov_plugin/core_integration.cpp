// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "behavior/ov_plugin/query_model.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_OVClassModelTestP, OVClassModelTestP,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_GPU)));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassModelOptionalTestP, OVClassModelOptionalTestP,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_GPU)));

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

// OV Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest, OVClassQueryModelTest, ::testing::Values("GPU"));

}  // namespace

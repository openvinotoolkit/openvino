// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "intel_npu/config/common.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

using namespace ov::test::behavior;

namespace {

// Several devices case
INSTANTIATE_TEST_SUITE_P(compatibility_nightly_BehaviorTests_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestCompileModel,
                         ::testing::Values(std::vector<std::string>(
                             {std::string(ov::test::utils::DEVICE_NPU) + "." +
                              removeDeviceNameOnlyID(ov::test::utils::getTestsPlatformFromEnvironmentOr("3720"))})),
                         (ov::test::utils::appendPlatformTypeTestName<OVClassSeveralDevicesTestCompileModel>));

INSTANTIATE_TEST_SUITE_P(compatibility_nightly_BehaviorTests_OVClassModelOptionalTestP,
                         OVClassModelOptionalTestP,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         (ov::test::utils::appendPlatformTypeTestName<OVClassModelOptionalTestP>));

INSTANTIATE_TEST_SUITE_P(compatibility_nightly_BehaviorTests_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestQueryModel,
                         ::testing::Values(std::vector<std::string>(
                             {std::string(ov::test::utils::DEVICE_NPU) + "." +
                                  removeDeviceNameOnlyID(ov::test::utils::getTestsPlatformFromEnvironmentOr("3720")),
                              std::string(ov::test::utils::DEVICE_NPU) + "." +
                                  removeDeviceNameOnlyID(ov::test::utils::getTestsPlatformFromEnvironmentOr("3720"))})),
                         (ov::test::utils::appendPlatformTypeTestName<OVClassSeveralDevicesTestQueryModel>));

}  // namespace

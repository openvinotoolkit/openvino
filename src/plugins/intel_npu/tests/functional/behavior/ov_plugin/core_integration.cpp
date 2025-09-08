// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/ov_plugin/core_integration.hpp"

using namespace ov::test::behavior;

namespace {

const char* NPU_PLUGIN_LIB_NAME = "openvino_intel_npu_plugin";

std::vector<std::string> devices = {
    std::string(ov::test::utils::DEVICE_NPU),
};

std::pair<std::string, std::string> plugins[] = {
    std::make_pair(std::string(NPU_PLUGIN_LIB_NAME), std::string(ov::test::utils::DEVICE_NPU)),
};

namespace OVClassBasicTestName {
static std::string getTestCaseName(testing::TestParamInfo<std::pair<std::string, std::string>> obj) {
    std::ostringstream result;
    result << "OVClassBasicTestName_" << obj.param.first << "_" << obj.param.second;
    result << "_targetDevice=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    return result.str();
}
}  // namespace OVClassBasicTestName

namespace OVClassNetworkTestName {
static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    result << "OVClassNetworkTestName_" << obj.param;
    result << "_targetDevice=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    return result.str();
}
}  // namespace OVClassNetworkTestName

const std::vector<ov::AnyMap> configs = {{}};

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_OVClassBasicTestP,
                         OVClassBasicTestPNPU,
                         ::testing::ValuesIn(plugins),
                         OVClassBasicTestName::getTestCaseName);
#endif

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_OVClassNetworkTestP,
                         OVClassNetworkTestPNPU,
                         ::testing::Combine(::testing::ValuesIn(devices), ::testing::ValuesIn(configs)),
                         OVClassNetworkTestPNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_OVClassLoadNetworkTest,
                         OVClassLoadNetworkTestNPU,
                         ::testing::Combine(::testing::ValuesIn(devices), ::testing::ValuesIn(configs)),
                         OVClassLoadNetworkTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_BehaviorTests_OVClassGetMetricTest,
                         OVClassGetMetricAndPrintNoThrow,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         OVClassNetworkTestName::getTestCaseName);

}  // namespace

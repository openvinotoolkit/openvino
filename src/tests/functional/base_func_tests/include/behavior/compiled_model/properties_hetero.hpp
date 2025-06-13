// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <base/ov_behavior_test_utils.hpp>

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>

#endif

namespace ov {
namespace test {
namespace behavior {

class OVClassHeteroCompiledModelGetMetricTest :
        public OVClassNetworkTest,
        public ::testing::WithParamInterface<std::string>,
        public OVCompiledNetworkTestBase {
protected:
    std::string heteroDeviceName;
    void SetCpuAffinity(ov::Core& core, std::vector<std::string>& expectedTargets);

public:
    void SetUp() override {
        target_device = GetParam();
        heteroDeviceName = ov::test::utils::DEVICE_HETERO + std::string(":") + target_device;
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
    }
};
using OVClassHeteroCompiledModelGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassHeteroCompiledModelGetMetricTest;
using OVClassHeteroCompiledModelGetMetricTest_TARGET_FALLBACK = OVClassHeteroCompiledModelGetMetricTest;
using OVClassHeteroCompiledModelGetMetricTest_EXEC_DEVICES = OVClassHeteroCompiledModelGetMetricTest;

}  // namespace behavior
}  // namespace test
}  // namespace ov

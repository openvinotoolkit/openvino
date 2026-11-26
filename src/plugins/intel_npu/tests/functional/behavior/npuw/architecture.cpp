// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NPUW picks its device-specific defaults (NPU_TILES, compilation params, fused HFA, ...) from the value that the
// NPU plugin reports for ov::device::architecture, queried both from the plugin and through the core
// (core->get_property("NPU", ov::device::architecture)). The unit tests cover what NPUW does with every value using
// a mocked plugin; this test pins the other end of the contract on real hardware: the value reported by the real
// device must be the platform the test machine is declared to be.
//
// The reference is IE_NPU_TESTS_PLATFORM, which is independent of the plugin (the plugin derives the architecture
// from the Level Zero device ID, see ZeroDevice::getName()). No platform list is kept here on purpose: for a new
// platform it is enough to run on the new machine with IE_NPU_TESTS_PLATFORM set. A device ID missing from
// ZeroDevice::getName() is reported as AUTO_DETECT and fails this test.

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "common/npu_test_env_cfg.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

TEST(NPUWArchitectureFunctionalTest, ReportedArchitectureMatchesTestPlatform) {
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "NPU") == devices.end()) {
        GTEST_SKIP() << "No NPU device available.";
    }

    const auto expected = ov::test::utils::getTestPlatform();
    if (expected.empty()) {
        GTEST_SKIP() << "IE_NPU_TESTS_PLATFORM is not set: there is nothing to compare the architecture with.";
    }

    std::string architecture;
    ASSERT_NO_THROW(architecture = core.get_property("NPU", ov::device::architecture));

    EXPECT_EQ(architecture, expected)
        << "Device '" << core.get_property("NPU", ov::device::full_name) << "' reports architecture '" << architecture
        << "', while the test machine is declared as '" << expected
        << "' (IE_NPU_TESTS_PLATFORM). If the reported value is AUTO_DETECT, the device ID "
           "is probably missing from ZeroDevice::getName().";
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/device_helpers.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "intel_npu/npu_private_properties.hpp"
#include "npu/fake_backend_utils.hpp"

namespace {

using ov::test::npu::FakeEngineBackend;

ov::SoPtr<::intel_npu::IEngineBackend> makeBackend(std::vector<std::string> deviceNames) {
    return ov::SoPtr<::intel_npu::IEngineBackend>(std::make_shared<FakeEngineBackend>(std::move(deviceNames)));
}

const std::string NPU4000{ov::intel_npu::Platform::NPU4000};
const std::string NPU5010{ov::intel_npu::Platform::NPU5010};

using ::intel_npu::utils::getDeviceArchitecture;

TEST(DeviceHelpersGetDeviceArchitectureTests, ReturnsEmptyWithoutBackend) {
    EXPECT_EQ(getDeviceArchitecture({nullptr}, ""), "");
    EXPECT_EQ(getDeviceArchitecture({nullptr}, NPU4000), "");
}

TEST(DeviceHelpersGetDeviceArchitectureTests, ReturnsEmptyWhenBackendHasNoDevices) {
    const auto backend = makeBackend({});

    EXPECT_NO_THROW(EXPECT_EQ(getDeviceArchitecture(backend, ""), ""));
    EXPECT_NO_THROW(EXPECT_EQ(getDeviceArchitecture(backend, NPU4000), ""));
}

TEST(DeviceHelpersGetDeviceArchitectureTests, EmptyDeviceIdUsesFirstDevice) {
    const auto backend = makeBackend({NPU4000, NPU5010});

    EXPECT_EQ(getDeviceArchitecture(backend, ""), NPU4000);
}

TEST(DeviceHelpersGetDeviceArchitectureTests, DeviceIdAsNameReturnsMatchingDevice) {
    const auto backend = makeBackend({NPU4000, NPU5010});

    EXPECT_EQ(getDeviceArchitecture(backend, NPU5010), NPU5010);
}

TEST(DeviceHelpersGetDeviceArchitectureTests, DeviceIdAsIndexReturnsDeviceNameNotIndex) {
    const auto backend = makeBackend({NPU4000, NPU5010});

    EXPECT_EQ(getDeviceArchitecture(backend, "0"), NPU4000);
    EXPECT_EQ(getDeviceArchitecture(backend, "1"), NPU5010);
}

TEST(DeviceHelpersGetDeviceArchitectureTests, ReturnsEmptyForUnknownDevice) {
    const auto backend = makeBackend({NPU4000});

    EXPECT_NO_THROW(EXPECT_EQ(getDeviceArchitecture(backend, "UNKNOWN"), ""));
    EXPECT_NO_THROW(EXPECT_EQ(getDeviceArchitecture(backend, "7"), ""));
}

}  // namespace

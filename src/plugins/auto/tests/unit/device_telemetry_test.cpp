// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "utils/device_telemetry.hpp"

using namespace ov::auto_plugin;

// Parameters: OpenVINO device name, expected telemetry metric key.
using DeviceMonitorKeyParams = std::tuple<std::string,  // device name
                                          std::string,  // device type
                                          std::string   // expected metric key
                                          >;

class DeviceMonitorKeyTest : public ::testing::TestWithParam<DeviceMonitorKeyParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DeviceMonitorKeyParams>& obj) {
        const auto& [device_name, device_type, expected_key] = obj.param;
        std::ostringstream result;
        std::string sanitized = device_name.empty() ? "empty" : device_name;
        std::replace(sanitized.begin(), sanitized.end(), '.', '_');
        result << "device_" << sanitized << "_type_" << (device_type.empty() ? "none" : device_type) << "_key_"
               << (expected_key.empty() ? "none" : expected_key);
        return result.str();
    }
};

TEST_P(DeviceMonitorKeyTest, maps_device_name_to_metric_key) {
    const auto& [device_name, device_type, expected_key] = GetParam();
    EXPECT_EQ(device_monitor::device_to_metric_key(device_name, device_type), expected_key);
}

const std::vector<DeviceMonitorKeyParams> deviceMonitorKeyConfigs = {
    DeviceMonitorKeyParams{"CPU", "", "CPUUtilization"},
    DeviceMonitorKeyParams{"GPU", "integrated", "IGPUUtilization"},
    DeviceMonitorKeyParams{"GPU", "discrete", "DGPUUtilization"},
    DeviceMonitorKeyParams{"GPU.0", "integrated", "IGPUUtilization"},
    DeviceMonitorKeyParams{"GPU.1", "discrete", "DGPUUtilization"},
    DeviceMonitorKeyParams{"GPU", "", ""},
    DeviceMonitorKeyParams{"NPU", "", "NPUUtilization"},
    DeviceMonitorKeyParams{"UNKNOWN", "", ""},
    DeviceMonitorKeyParams{"", "", ""}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         DeviceMonitorKeyTest,
                         ::testing::ValuesIn(deviceMonitorKeyConfigs),
                         DeviceMonitorKeyTest::getTestCaseName);

// TelemetryClient::utilization must never throw and must return a value within
// [0.0, 100.0] when available, or std::nullopt otherwise. On builds without the
// telemetry backend it consistently returns std::nullopt.
TEST(DeviceMonitorTest, telemetry_client_utilization_is_safe) {
    device_monitor::TelemetryClient client;
    std::optional<float> utilization;
    ASSERT_NO_THROW(utilization = client.utilization("CPU"));
    if (utilization.has_value()) {
        EXPECT_GE(utilization.value(), 0.0f);
        EXPECT_LE(utilization.value(), 100.0f);
    }
}

TEST(DeviceMonitorTest, telemetry_client_unknown_device_returns_nullopt) {
    device_monitor::TelemetryClient client;
    std::optional<float> utilization;
    ASSERT_NO_THROW(utilization = client.utilization("UNKNOWN_DEVICE"));
    EXPECT_FALSE(utilization.has_value());
}

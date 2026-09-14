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

TEST(DeviceMonitorTest, low_power_mode_gear_mapping_matches_expected_policy) {
    // Gears 1-3 request low-latency/performance operation (perf_curve_table); only
    // gears 4-7 request low power operation (low_power_device).
    EXPECT_FALSE(device_monitor::is_low_power_gear(-1));
    EXPECT_FALSE(device_monitor::is_low_power_gear(0));
    EXPECT_FALSE(device_monitor::is_low_power_gear(1));
    EXPECT_FALSE(device_monitor::is_low_power_gear(2));
    EXPECT_FALSE(device_monitor::is_low_power_gear(3));
    EXPECT_TRUE(device_monitor::is_low_power_gear(4));
    EXPECT_TRUE(device_monitor::is_low_power_gear(7));
    // Gears above the EPO-defined range are not low power.
    EXPECT_FALSE(device_monitor::is_low_power_gear(8));
    EXPECT_FALSE(device_monitor::is_low_power_gear(100));
}

TEST(DeviceMonitorTest, valid_gear_range_matches_epo_specification) {
    // EPO defines gears 1-7; anything outside that range is not a valid gear.
    EXPECT_FALSE(device_monitor::is_valid_gear(-1));
    EXPECT_FALSE(device_monitor::is_valid_gear(0));
    EXPECT_TRUE(device_monitor::is_valid_gear(1));
    EXPECT_TRUE(device_monitor::is_valid_gear(7));
    EXPECT_FALSE(device_monitor::is_valid_gear(8));
    EXPECT_FALSE(device_monitor::is_valid_gear(100));
}

TEST(DeviceMonitorTest, telemetry_client_low_power_mode_is_safe) {
    device_monitor::TelemetryClient client;
    std::optional<bool> low_power_mode;
    ASSERT_NO_THROW(low_power_mode = client.is_low_power_mode());
#ifndef OV_AUTO_ENABLE_IPF
    EXPECT_FALSE(low_power_mode.has_value());
#endif
}

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

#ifdef OV_AUTO_ENABLE_IPF
TEST(DeviceMonitorTest, parse_utilization_uses_gpu_fallback_for_igpu) {
    const std::string aiselector_json = R"({
        "Performance": {
            "GPUUtilization": 4.63
        },
        "Status": "Online"
    })";

    const auto utilization = device_monitor::parse_utilization_from_aiselector_json_for_test(aiselector_json,
                                                                                              "GPU",
                                                                                              "integrated");
    ASSERT_TRUE(utilization.has_value());
    EXPECT_FLOAT_EQ(utilization.value(), 4.63f);
}

TEST(DeviceMonitorTest, parse_utilization_returns_nullopt_when_igpu_keys_missing) {
    const std::string aiselector_json = R"({
        "Performance": {
            "CPUUtilization": 20.83
        },
        "Status": "Online"
    })";

    const auto utilization = device_monitor::parse_utilization_from_aiselector_json_for_test(aiselector_json,
                                                                                              "GPU",
                                                                                              "integrated");
    EXPECT_FALSE(utilization.has_value());
}
#endif

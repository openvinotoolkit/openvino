// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "utils/device_telemetry.hpp"
#include "utils/ipf_client.hpp"

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

// TelemetryClient::utilizations must never throw and must return values within
// [0.0, 100.0] when available. On builds without the telemetry backend it consistently
// returns an empty map.
TEST(DeviceMonitorTest, telemetry_client_utilizations_is_safe) {
    device_monitor::TelemetryClient client;
    std::unordered_map<std::string, float> utilizations;
    ASSERT_NO_THROW(utilizations = client.utilizations({{"CPU", ""}}));
    for (const auto& [device_name, utilization] : utilizations) {
        EXPECT_GE(utilization, 0.0f);
        EXPECT_LE(utilization, 100.0f);
    }
}

TEST(DeviceMonitorTest, telemetry_client_unknown_device_is_excluded) {
    device_monitor::TelemetryClient client;
    std::unordered_map<std::string, float> utilizations;
    ASSERT_NO_THROW(utilizations = client.utilizations({{"UNKNOWN_DEVICE", ""}}));
    EXPECT_EQ(utilizations.count("UNKNOWN_DEVICE"), 0u);
}

#ifdef OV_AUTO_ENABLE_IPF
class MockIpfClient : public device_monitor::IIpfClient {
public:
    MOCK_METHOD(bool, is_valid, (), (const, override));
    MOCK_METHOD(std::string, get_node, (const std::string&), (override));
    MOCK_METHOD(std::string, get_value, (const std::string&), (override));
    MOCK_METHOD(bool, register_event, (const std::string&, device_monitor::IpfEventCallback), (override));
    MOCK_METHOD(void, unregister_event, (const std::string&), (override));
};

TEST(DeviceMonitorTest, telemetry_client_reports_unavailable_when_ipf_invalid) {
    auto mock = std::make_unique<::testing::NiceMock<MockIpfClient>>();
    EXPECT_CALL(*mock, is_valid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mock, get_node(::testing::_)).Times(0);
    EXPECT_CALL(*mock, register_event(::testing::_, ::testing::_)).Times(0);

    device_monitor::TelemetryClient client(std::move(mock));
    EXPECT_FALSE(client.is_available());
    EXPECT_TRUE(client.utilizations({{"CPU", ""}}).empty());
    EXPECT_FALSE(client.is_low_power_mode().has_value());
}

TEST(DeviceMonitorTest, telemetry_client_utilizations_returns_parsed_value_for_cpu) {
    auto mock = std::make_unique<::testing::NiceMock<MockIpfClient>>();
    EXPECT_CALL(*mock, is_valid()).WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*mock, get_node(std::string("Platform.Features.AISelector")))
        .WillOnce(::testing::Return(R"({"Performance": {"CPUUtilization": 42.5}, "Status": "Online"})"));

    device_monitor::TelemetryClient client(std::move(mock));
    const auto utilizations = client.utilizations({{"CPU", ""}});
    ASSERT_EQ(utilizations.count("CPU"), 1u);
    EXPECT_FLOAT_EQ(utilizations.at("CPU"), 42.5f);
}

TEST(DeviceMonitorTest, telemetry_client_utilizations_empty_when_query_fails) {
    auto mock = std::make_unique<::testing::NiceMock<MockIpfClient>>();
    EXPECT_CALL(*mock, is_valid()).WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*mock, get_node(::testing::_)).WillOnce(::testing::Return(""));

    device_monitor::TelemetryClient client(std::move(mock));
    EXPECT_TRUE(client.utilizations({{"CPU", ""}}).empty());
}

TEST(DeviceMonitorTest, telemetry_client_is_low_power_mode_reflects_initial_gear) {
    auto mock = std::make_unique<::testing::NiceMock<MockIpfClient>>();
    EXPECT_CALL(*mock, is_valid()).WillRepeatedly(::testing::Return(true));
    // Matches both the DTT status and EPO status paths; either being unreachable disables gear tracking.
    ON_CALL(*mock, get_value(::testing::HasSubstr("Status"))).WillByDefault(::testing::Return(R"("Enabled")"));
    ON_CALL(*mock, get_value(::testing::HasSubstr("Version"))).WillByDefault(::testing::Return(R"("1.2.3")"));
    // Gear 5 is within the EPO low-power range [4,7].
    ON_CALL(*mock, get_value(::testing::HasSubstr("CurrentGear"))).WillByDefault(::testing::Return(R"("5")"));
    ON_CALL(*mock, register_event(::testing::_, ::testing::_)).WillByDefault(::testing::Return(true));

    device_monitor::TelemetryClient client(std::move(mock));
    const auto low_power_mode = client.is_low_power_mode();
    ASSERT_TRUE(low_power_mode.has_value());
    EXPECT_TRUE(low_power_mode.value());
}

TEST(DeviceMonitorTest, telemetry_client_is_low_power_mode_updates_on_gear_changed_event) {
    auto mock = std::make_unique<::testing::NiceMock<MockIpfClient>>();
    device_monitor::IpfEventCallback captured_callback;
    EXPECT_CALL(*mock, is_valid()).WillRepeatedly(::testing::Return(true));
    ON_CALL(*mock, get_value(::testing::HasSubstr("Status"))).WillByDefault(::testing::Return(R"("Enabled")"));
    ON_CALL(*mock, get_value(::testing::HasSubstr("Version"))).WillByDefault(::testing::Return(R"("1.2.3")"));
    // Gear 1 is valid (EPO range [1,7]) but not a low-power gear.
    ON_CALL(*mock, get_value(::testing::HasSubstr("CurrentGear"))).WillByDefault(::testing::Return(R"("1")"));
    EXPECT_CALL(*mock, register_event(::testing::_, ::testing::_))
        .WillOnce(::testing::DoAll(::testing::SaveArg<1>(&captured_callback), ::testing::Return(true)));

    device_monitor::TelemetryClient client(std::move(mock));
    const auto initial_low_power = client.is_low_power_mode();
    ASSERT_TRUE(initial_low_power.has_value());
    EXPECT_FALSE(initial_low_power.value());

    ASSERT_TRUE(static_cast<bool>(captured_callback));
    // Gear 5 is within the EPO low-power range [4,7].
    captured_callback(R"({"OnEpoGearChanged": "5"})");
    const auto updated_low_power = client.is_low_power_mode();
    ASSERT_TRUE(updated_low_power.has_value());
    EXPECT_TRUE(updated_low_power.value());
}

TEST(DeviceMonitorTest, telemetry_client_unregisters_event_on_destruction_when_registered) {
    auto mock = std::make_unique<::testing::NiceMock<MockIpfClient>>();
    EXPECT_CALL(*mock, is_valid()).WillRepeatedly(::testing::Return(true));
    ON_CALL(*mock, get_value(::testing::HasSubstr("Status"))).WillByDefault(::testing::Return(R"("Enabled")"));
    ON_CALL(*mock, register_event(::testing::_, ::testing::_)).WillByDefault(::testing::Return(true));
    EXPECT_CALL(*mock, unregister_event(::testing::_)).Times(1);

    {
        device_monitor::TelemetryClient client(std::move(mock));
        client.is_low_power_mode();
    }
}

TEST(DeviceMonitorTest, utilization_values_are_parsed_from_one_snapshot) {
    const std::string snapshot = R"({
        "Performance": {
            "CPUUtilization": 12.5,
            "NPUUtilization": 37.5
        },
        "Status": "Online"
    })";

    const auto utilizations = device_monitor::utilization_from_snapshot(
        snapshot,
        std::vector<std::pair<std::string, std::string>>{{"CPU", ""}, {"NPU", ""}});

    ASSERT_EQ(utilizations.count("CPU"), 1u);
    ASSERT_EQ(utilizations.count("NPU"), 1u);
    EXPECT_FLOAT_EQ(utilizations.at("CPU"), 12.5f);
    EXPECT_FLOAT_EQ(utilizations.at("NPU"), 37.5f);
}
#endif

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

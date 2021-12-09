// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include "behavior_test_plugin.h"

class VPUGetMetric : public testing::Test {
protected:
    InferenceEngine::Core ie;

    std::vector<std::string> getAvailableDevices() {
        auto result = Parameter{};
        result = ie.GetMetric("MYRIAD", METRIC_KEY(AVAILABLE_DEVICES));

        return result.as<std::vector<std::string>>();
    }

    ExecutableNetwork loadNetworkOnDevice(const std::string &deviceName) {
        auto network = ie.ReadNetwork(convReluNormPoolFcModelFP16.model_xml_str,
                                      convReluNormPoolFcModelFP16.weights_blob);

        return ie.LoadNetwork(network, deviceName);
    }
};

TEST_F(VPUGetMetric, smoke_GetThermalStatsFromNetwork) {
    const auto exe_network = loadNetworkOnDevice("MYRIAD");

    auto result = Parameter{};
    ASSERT_NO_THROW(result = exe_network.GetMetric(METRIC_KEY(DEVICE_THERMAL)));

    ASSERT_FALSE(result.empty());
    ASSERT_GT(result.as<float>(), 0);
}

TEST_F(VPUGetMetric, smoke_GetThermalStatsFromPlugin) {
    std::vector<std::string> availableDevices;
    ASSERT_NO_THROW(availableDevices = getAvailableDevices());
    ASSERT_TRUE(!availableDevices.empty());

    for (const auto &availableDevice : availableDevices) {
        const auto deviceName = "MYRIAD." + availableDevice;
        ASSERT_NO_THROW(loadNetworkOnDevice(deviceName));

        auto result = Parameter{};
        ASSERT_NO_THROW(result = ie.GetMetric(deviceName, METRIC_KEY(DEVICE_THERMAL)));

        ASSERT_FALSE(result.empty());
        ASSERT_GT(result.as<float>(), 0.f);
    }
}

TEST_F(VPUGetMetric, smoke_ThermalStatsFromPluginWithIncorrectID) {
    std::vector<std::string> availableDevices;
    ASSERT_NO_THROW(availableDevices = getAvailableDevices());
    ASSERT_TRUE(!availableDevices.empty());

    // Load network with correct device to fill the device pool.
    const auto deviceName = "MYRIAD." + availableDevices.front();
    ASSERT_NO_THROW(loadNetworkOnDevice(deviceName));

    // Try to get DEVICE_THERMAL metric for a device with incorrect name.
    // This should result in an exception.
    const auto incorrectDeviceName = "MYRIAD.incorrect_device";
    auto result = Parameter{};
    ASSERT_NO_THROW(result = ie.GetMetric(incorrectDeviceName, METRIC_KEY(DEVICE_THERMAL)));
    ASSERT_TRUE(result.empty());
}

TEST_F(VPUGetMetric, smoke_ThermalStatsFromPluginWithoutLoadedNetwork) {
    std::vector<std::string> availableDevices;
    ASSERT_NO_THROW(availableDevices = getAvailableDevices());
    ASSERT_TRUE(!availableDevices.empty());

    // Try to get DEVICE_THERMAL metric for a device on which the network is not loaded.
    // This should result in an exception.
    const auto deviceName = "MYRIAD." + availableDevices.front();
    auto result = Parameter{};
    ASSERT_NO_THROW(result = ie.GetMetric(deviceName, METRIC_KEY(DEVICE_THERMAL)));
    ASSERT_TRUE(result.empty());
}

TEST_F(VPUGetMetric, smoke_MyriadGetFullDeviceName) {
    std::vector<std::string> availableDevices;
    ASSERT_NO_THROW(availableDevices = getAvailableDevices());
    ASSERT_TRUE(!availableDevices.empty());

    auto result = Parameter{};
    for (size_t i = 0; i < availableDevices.size(); ++i) {
        const auto deviceName = "MYRIAD." + availableDevices[i];
        ASSERT_NO_THROW(result = ie.GetMetric(deviceName, METRIC_KEY(FULL_DEVICE_NAME)));
        auto act_res = result.as<std::string>();
        ASSERT_TRUE(!act_res.empty());
    }
}

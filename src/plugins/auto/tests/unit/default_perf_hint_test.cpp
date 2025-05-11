// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
using namespace ov::mock_auto_plugin;

using ConfigParams = std::tuple<std::string,               // virtual device name to load network
                                std::vector<std::string>,  // hardware device name to expect loading network on
                                ov::AnyMap>;               // secondary property setting to device

static std::vector<ConfigParams> testConfigs;

class AutoDefaultPerfHintTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string deviceName;
        std::vector<std::string> targetDevices;
        ov::AnyMap deviceConfigs;
        std::tie(deviceName, targetDevices, deviceConfigs) = obj.param;
        std::ostringstream result;
        result << deviceName;
        result << "_loadnetwork_to_";
        for (auto& device : targetDevices) {
            result << device << "_";
        }
        for (auto& item : deviceConfigs) {
            result << item.first << "_" << item.second.as<std::string>() << "_";
        }
        auto name = result.str();
        name.pop_back();
        return name;
    }

    static std::vector<ConfigParams> CreateNumStreamsAndDefaultPerfHintTestConfigs() {
        testConfigs.clear();
        testConfigs.push_back(
            ConfigParams{"AUTO", {"CPU"}, {{"MULTI_DEVICE_PRIORITIES", "CPU"}}});  // CPU: get default_hint:lantency
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU"}}});  // CPU: no perf_hint
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get default_hint:lantency GPU:get default_hint:lantency
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"},
                          {"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get default_hint:lantency GPU:get default_hint:lantency
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"CPU", "GPU"},
            {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:3}}"},
             {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});  // CPU: as helper, get default_hint:lantency GPU:no perf_hint
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:5}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint
        testConfigs.push_back(
            ConfigParams{"AUTO", {"GPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU"}}});  // GPU: get default_hint:lantency
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"GPU"},
                                           {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:3}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "GPU"}}});  // GPU: no perf_hint

        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: get default_hint:tput
        testConfigs.push_back(
            ConfigParams{"MULTI:CPU,GPU",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"},
                          {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint  GPU: get default_hint:tput
        testConfigs.push_back(
            ConfigParams{"MULTI:CPU,GPU",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:3}}"},
                          {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: no perf_hint
        testConfigs.push_back(
            ConfigParams{"MULTI:CPU,GPU",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3},{GPU:{NUM_STREAMS:3}}"},
                          {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint  GPU: no perf_hint
        return testConfigs;
    }

    static std::vector<ConfigParams> CreatePerfHintAndDefaultPerfHintTestConfigs() {
        testConfigs.clear();
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"DEVICE_PROPERTIES", "{CPU:{PERFORMANCE_HINT:THROUGHPUT}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU"}}});  // CPU: get perf_hint:tput
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{PERFORMANCE_HINT:THROUGHPUT}}"},
                          {"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get perf_hint:lantency GPU:get default_hint:lantency
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"CPU", "GPU"},
            {{"DEVICE_PROPERTIES", "{CPU:{PERFORMANCE_HINT:THROUGHPUT},GPU:{PERFORMANCE_HINT:THROUGHPUT}}"},
             {"MULTI_DEVICE_PRIORITIES",
              "GPU,CPU"}}});  // CPU: as helper, get perf_hint:lantency GPU:get perf_hint:tput
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"DEVICE_PROPERTIES", "{CPU:{PERFORMANCE_HINT:THROUGHPUT}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get perf_hint:tput
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"GPU"},
                                           {{"DEVICE_PROPERTIES", "{GPU:{PERFORMANCE_HINT:THROUGHPUT}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "GPU"}}});  // GPU: get perf_hint:tput

        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"DEVICE_PROPERTIES", "{CPU:{PERFORMANCE_HINT:LATENCY}}"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get perf_hint:latency  GPU: get default_hint:tput
        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"DEVICE_PROPERTIES", "{GPU:{PERFORMANCE_HINT:THROUGHPUT}}"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: get perf_hint:tput
        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"DEVICE_PROPERTIES", "{CPU:{PERFORMANCE_HINT:THROUGHPUT},GPU:{PERFORMANCE_HINT:THROUGHPUT}}"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get perf_hint:tput  GPU: get perf_hint:tput
        return testConfigs;
    }

    static std::vector<ConfigParams> CreateSecPropAndDefaultPerfHintTestConfigs() {
        testConfigs.clear();
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"DEVICE_PROPERTIES", "{CPU:{ALLOW_AUTO_BATCHING:TRUE}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU"}}});  // CPU: no perf_hint
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{ALLOW_AUTO_BATCHING:TRUE}}"},
                          {"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get perf_hint:lantency GPU:get default_hint:lantency
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"CPU", "GPU"},
            {{"DEVICE_PROPERTIES", "{CPU:{ALLOW_AUTO_BATCHING:TRUE},GPU:{ALLOW_AUTO_BATCHING:TRUE}}"},
             {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});  // CPU: as helper, get perf_hint:lantency GPU:no perf_hint
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"DEVICE_PROPERTIES", "{CPU:{ALLOW_AUTO_BATCHING:FALSE}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"GPU"},
                                           {{"DEVICE_PROPERTIES", "{GPU:{ALLOW_AUTO_BATCHING:FALSE}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "GPU"}}});  // GPU: no perf_hint

        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"DEVICE_PROPERTIES", "{CPU:{ALLOW_AUTO_BATCHING:FALSE}}"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: get default_hint:tput
        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"DEVICE_PROPERTIES", "{GPU:{ALLOW_AUTO_BATCHING:FALSE}}"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: get default_hint:tput
        testConfigs.push_back(
            ConfigParams{"MULTI:CPU,GPU",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{ALLOW_AUTO_BATCHING:TRUE},GPU:{ALLOW_AUTO_BATCHING:FALSE}}"},
                          {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint GPU: get default_hint:tput
        return testConfigs;
    }

    void SetUp() override {
        std::vector<std::string> availableDevs = {"CPU", "GPU"};
        ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));
        std::vector<std::string> deviceIDs = {};
        ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::available_devices.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(deviceIDs));

        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("CPU")),
                              _))
            .WillByDefault(Return(mockExeNetwork));

        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq("GPU")),
                              _))
            .WillByDefault(Return(mockExeNetworkActual));
    }
};

using NumStreamsAndDefaultPerfHintMockTest = AutoDefaultPerfHintTest;
using PerHintAndDefaultPerfHintMockTest = AutoDefaultPerfHintTest;
using SecPropAndDefaultPerfHintMockTest = AutoDefaultPerfHintTest;

TEST_P(NumStreamsAndDefaultPerfHintMockTest, NumStreamsAndDefaultPerfHintTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    ov::AnyMap config;
    bool bIsAuto = false;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos) {
        bIsAuto = true;
        plugin->set_device_name("AUTO");
    }

    if (device.find("MULTI") != std::string::npos)
        plugin->set_device_name("MULTI");

    for (auto& deviceName : targetDevices) {
        bool isCPUHelper = false;
        if (deviceName.find("CPU") != std::string::npos && bIsAuto) {
            auto item = config.find(ov::device::priorities.name());
            if (item != config.end() && item->second.as<std::string>().find("GPU,CPU") != std::string::npos) {
                isCPUHelper = true;
            }
        }
        std::string HW_PerfHint;
        if (isCPUHelper) {
            // if it is CPU Helper, CPU should keep perf hint to LATENCY.
            HW_PerfHint = "LATENCY";
        } else {
            // HW default perf_hint
            HW_PerfHint = bIsAuto ? "LATENCY" : "THROUGHPUT";
        }

        auto item = config.find(ov::device::properties.name());
        ov::AnyMap deviceConfigs;
        if (item != config.end()) {
            ov::AnyMap devicesProperties;
            std::stringstream strConfigs(item->second.as<std::string>());
            // Parse the device properties to common property into deviceConfigs.
            ov::util::Read<ov::AnyMap>{}(strConfigs, devicesProperties);
            auto it = devicesProperties.find(deviceName);
            if (it != devicesProperties.end()) {
                std::stringstream strConfigs(it->second.as<std::string>());
                ov::util::Read<ov::AnyMap>{}(strConfigs, deviceConfigs);
            }
        }
        bool bNumStreams = deviceConfigs.find(ov::num_streams.name()) != deviceConfigs.end() ? true : false;
        if (bNumStreams && !isCPUHelper) {
            // do not pass default perf_hint to HW
            HW_PerfHint = "No PERFORMANCE_HINT";
        }
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(deviceName),
                                  ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint(HW_PerfHint))))
            .Times(1);
    }

    OV_ASSERT_NO_THROW(plugin->compile_model(model, config));
}

INSTANTIATE_TEST_SUITE_P(
    smoke_AutoMultiMock_NumStreamsAndDefaultPerfHintToHWTest,
    NumStreamsAndDefaultPerfHintMockTest,
    ::testing::ValuesIn(NumStreamsAndDefaultPerfHintMockTest::CreateNumStreamsAndDefaultPerfHintTestConfigs()),
    NumStreamsAndDefaultPerfHintMockTest::getTestCaseName);

TEST_P(PerHintAndDefaultPerfHintMockTest, PerfHintAndDefaultPerfHintTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    ov::AnyMap config;
    bool bIsAuto = false;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos) {
        bIsAuto = true;
        plugin->set_device_name("AUTO");
    }

    if (device.find("MULTI") != std::string::npos)
        plugin->set_device_name("MULTI");

    for (auto& deviceName : targetDevices) {
        bool isCPUHelper = false;
        if (deviceName.find("CPU") != std::string::npos && bIsAuto) {
            auto item = config.find(ov::device::priorities.name());
            if (item != config.end() && item->second.as<std::string>().find("GPU,CPU") != std::string::npos) {
                isCPUHelper = true;
            }
        }
        std::string HW_PerfHint;
        if (isCPUHelper) {
            // if it is CPU Helper, CPU should keep perf hint to LATENCY.
            HW_PerfHint = "LATENCY";
        } else {
            // HW default perf_hint
            HW_PerfHint = bIsAuto ? "LATENCY" : "THROUGHPUT";
        }
        auto item = config.find(ov::device::properties.name());
        ov::AnyMap deviceConfigs;
        if (item != config.end()) {
            ov::AnyMap devicesProperties;
            std::stringstream strConfigs(item->second.as<std::string>());
            // Parse the device properties to common property into deviceConfigs.
            ov::util::Read<ov::AnyMap>{}(strConfigs, devicesProperties);
            auto it = devicesProperties.find(deviceName);
            if (it != devicesProperties.end()) {
                std::stringstream strConfigs(it->second.as<std::string>());
                ov::util::Read<ov::AnyMap>{}(strConfigs, deviceConfigs);
            }
        }
        auto itor = deviceConfigs.find(ov::hint::performance_mode.name());
        if (itor != deviceConfigs.end() && !isCPUHelper) {
            HW_PerfHint = itor->second.as<std::string>();
        }
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(StrEq(deviceName)),
                                  ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint(HW_PerfHint))))
            .Times(1);
    }

    OV_ASSERT_NO_THROW(plugin->compile_model(model, config));
}

INSTANTIATE_TEST_SUITE_P(
    smoke_AutoMultiMock_PerHintAndDefaultPerfHintToHWTest,
    PerHintAndDefaultPerfHintMockTest,
    ::testing::ValuesIn(PerHintAndDefaultPerfHintMockTest::CreatePerfHintAndDefaultPerfHintTestConfigs()),
    PerHintAndDefaultPerfHintMockTest::getTestCaseName);

TEST_P(SecPropAndDefaultPerfHintMockTest, SecPropAndDefaultPerfHintTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    ov::AnyMap config;
    bool bIsAuto = false;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos) {
        bIsAuto = true;
        plugin->set_device_name("AUTO");
    }

    if (device.find("MULTI") != std::string::npos)
        plugin->set_device_name("MULTI");

    for (auto& deviceName : targetDevices) {
        bool isCPUHelper = false;
        if (deviceName.find("CPU") != std::string::npos && bIsAuto) {
            auto item = config.find(ov::device::priorities.name());
            if (item != config.end() && item->second.as<std::string>().find("GPU,CPU") != std::string::npos) {
                isCPUHelper = true;
            }
        }
        std::string HW_PerfHint;
        if (isCPUHelper) {
            // if it is CPU Helper, CPU should keep perf hint to LATENCY.
            HW_PerfHint = "LATENCY";
        } else {
            // HW default perf_hint
            HW_PerfHint = bIsAuto ? "LATENCY" : "THROUGHPUT";
        }

        auto item = config.find(ov::device::properties.name());
        if (item != config.end()) {
            ov::AnyMap deviceConfigs;
            ov::AnyMap devicesProperties;
            std::stringstream strConfigs(item->second.as<std::string>());
            // Parse the device properties to common property into deviceConfigs.
            ov::util::Read<ov::AnyMap>{}(strConfigs, devicesProperties);
            auto it = devicesProperties.find(deviceName);
            if (it != devicesProperties.end()) {
                // No default hint setting if device properties setting for hardware device.
                // do not pass default perf_hint to HW
                if (!isCPUHelper) {
                    HW_PerfHint = "No PERFORMANCE_HINT";
                }
            }
        }
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(StrEq(deviceName)),
                                  ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint(HW_PerfHint))))
            .Times(1);
    }

    OV_ASSERT_NO_THROW(plugin->compile_model(model, config));
}

INSTANTIATE_TEST_SUITE_P(
    smoke_AutoMultiMock_SecPropAndDefaultPerfHintToHWTest,
    SecPropAndDefaultPerfHintMockTest,
    ::testing::ValuesIn(SecPropAndDefaultPerfHintMockTest::CreateSecPropAndDefaultPerfHintTestConfigs()),
    SecPropAndDefaultPerfHintMockTest::getTestCaseName);
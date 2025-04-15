// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"

using namespace ov::mock_auto_plugin;
using Config = std::map<std::string, std::string>;
using testing::Throw;

const char igpuFullDeviceName[] = "Intel(R) Gen9 HD Graphics (iGPU)";
const char dgpuFullDeviceName[] = "Intel(R) Iris(R) Xe MAX Graphics (dGPU)";
const std::vector<std::string> availableDevsNoID = {"CPU", "GPU", "OTHER"};
using ConfigParams = std::tuple<std::string,                     // Priority devices
                                std::vector<DeviceInformation>,  // expect metaDevices
                                bool,                            // if throw exception
                                int                              // expected times of calling get_supported_property()
                                >;
class ParseMetaDeviceTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string priorityDevices;
        std::vector<DeviceInformation> metaDevices;
        bool throwException;
        int expectedTimes;
        std::tie(priorityDevices, metaDevices, throwException, expectedTimes) = obj.param;
        std::ostringstream result;
        result << "priorityDevices_" << priorityDevices;
        if (throwException) {
            result << "_throwException_true";
        } else {
            result << "_throwException_false";
        }
        result << "_expectedGetSupportedProperty_" << expectedTimes;
        return result.str();
    }

    void SetUp() override {
        ON_CALL(*core, get_supported_property(StrEq("INVALID_DEVICE"), _, _)).WillByDefault(ov::Throw(""));
        ON_CALL(*core, get_property(StrEq("GPU.2"), ov::supported_properties.name(), _)).WillByDefault(ov::Throw(""));
        ON_CALL(*plugin, parse_meta_devices)
            .WillByDefault([this](const std::string& priorityDevices, const ov::AnyMap& config) {
                return plugin->Plugin::parse_meta_devices(priorityDevices, config);
            });
        std::tie(priorityDevices, metaDevices, throwException, expectedTimes) = GetParam();
    }

    void compare(std::vector<DeviceInformation>& result, std::vector<DeviceInformation>& expect) {
        EXPECT_EQ(result.size(), expect.size());
        if (result.size() == expect.size()) {
            for (unsigned int i = 0; i < result.size(); i++) {
                EXPECT_EQ(result[i].device_name, expect[i].device_name);
                EXPECT_EQ(result[i].unique_name, expect[i].unique_name);
                EXPECT_EQ(result[i].num_requests_per_devices, expect[i].num_requests_per_devices);
                EXPECT_EQ(result[i].default_device_id, expect[i].default_device_id);
            }
        }
    }

    void compareDevicePriority(std::vector<DeviceInformation>& result, std::vector<DeviceInformation>& expect) {
        EXPECT_EQ(result.size(), expect.size());
        if (result.size() == expect.size()) {
            for (unsigned int i = 0; i < result.size(); i++) {
                EXPECT_EQ(result[i].device_priority, expect[i].device_priority);
            }
        }
    }

protected:
    // get Parameter
    std::string priorityDevices;
    std::vector<DeviceInformation> metaDevices;
    bool throwException;
    int expectedTimes;
};
using ParseMetaDeviceNoIDTest = ParseMetaDeviceTest;

TEST_P(ParseMetaDeviceTest, ParseMetaDevicesWithPriority) {
    EXPECT_CALL(*plugin, parse_meta_devices(_, _)).Times(1);
    EXPECT_CALL(*core, get_property(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(*core, get_available_devices()).Times(0);
    EXPECT_CALL(*core, get_supported_property(_, _, _)).Times(expectedTimes);
    if (throwException) {
        ASSERT_ANY_THROW(plugin->parse_meta_devices(priorityDevices, {}));
    } else {
        auto result = plugin->parse_meta_devices(priorityDevices, {ov::device::priorities(priorityDevices)});
        compare(result, metaDevices);
        compareDevicePriority(result, metaDevices);
    }
}

TEST_P(ParseMetaDeviceTest, ParseMetaDevicesNotWithPriority) {
    EXPECT_CALL(*plugin, parse_meta_devices(_, _)).Times(1 + !throwException);
    EXPECT_CALL(*core, get_property(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(*core, get_available_devices()).Times(0);
    if (throwException) {
        ASSERT_ANY_THROW(plugin->parse_meta_devices(priorityDevices, {}));
    } else {
        auto result = plugin->parse_meta_devices(priorityDevices, {});
        compare(result, metaDevices);
        for (unsigned int i = 0; i < result.size(); i++) {
            EXPECT_EQ(result[i].device_priority, 0);
        }
        auto result2 = plugin->parse_meta_devices(priorityDevices, {ov::device::priorities("")});
        compare(result2, metaDevices);
        for (unsigned int i = 0; i < result.size(); i++) {
            EXPECT_EQ(result2[i].device_priority, 0);
        }
    }
}

TEST_P(ParseMetaDeviceNoIDTest, ParseMetaDevices) {
    ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevsNoID));
    std::vector<std::string> deviceIDs = {};
    ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::available_devices.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(deviceIDs));
    EXPECT_CALL(*plugin, parse_meta_devices(_, _)).Times(1);
    EXPECT_CALL(*core, get_property(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(*core, get_available_devices()).Times(0);
    EXPECT_CALL(*core, get_supported_property(_, _, _)).Times(expectedTimes);
    if (throwException) {
        ASSERT_ANY_THROW(plugin->parse_meta_devices(priorityDevices, {}));
    } else {
        auto result = plugin->parse_meta_devices(priorityDevices, {ov::device::priorities(priorityDevices)});
        compare(result, metaDevices);
        compareDevicePriority(result, metaDevices);
    }
}
// ConfigParams details
// example
// ConfigParams {devicePriority, expect metaDevices, ifThrowException}

const std::vector<ConfigParams> testConfigs = {
    ConfigParams{"CPU,GPU.2,OTHER", {{"CPU", {}, -1, "", "CPU_", 0}, {"OTHER", {}, -1, "", "OTHER_", 2}}, false, 3},
    ConfigParams{"CPU,GPU,OTHER",
                 {{"CPU", {}, -1, "", "CPU_", 0},
                  {"GPU.0", {}, -1, "", std::string(igpuFullDeviceName) + "_0", 1},
                  {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1},
                  {"OTHER", {}, -1, "", "OTHER_", 2}},
                 false,
                 4},
    ConfigParams{"OTHER,GPU,CPU",
                 {{"OTHER", {}, -1, "", "OTHER_", 0},
                  {"GPU.0", {}, -1, "", std::string(igpuFullDeviceName) + "_0", 1},
                  {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1},
                  {"CPU", {}, -1, "", "CPU_", 2}},
                 false,
                 4},
    ConfigParams{"OTHER,GPU,INVALID_DEVICE",
                 {{"OTHER", {}, -1, "", "OTHER_", 0},
                  {"GPU.0", {}, -1, "", std::string(igpuFullDeviceName) + "_0", 1},
                  {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1}},
                 false,
                 3},
    ConfigParams{"CPU(1),GPU(2),OTHER(4)",
                 {{"CPU", {}, 1, "", "CPU_", 0},
                  {"GPU.0", {}, 2, "", std::string(igpuFullDeviceName) + "_0", 1},
                  {"GPU.1", {}, 2, "", std::string(dgpuFullDeviceName) + "_1", 1},
                  {"OTHER", {}, 4, "", "OTHER_", 2}},
                 false,
                 4},

    ConfigParams{"CPU(-1),GPU,OTHER", {}, true, 0},
    ConfigParams{"CPU(NA),GPU,OTHER", {}, true, 0},
    ConfigParams{"INVALID_DEVICE", {}, false, 0},
    ConfigParams{"INVALID_DEVICE,CPU", {{"CPU", {}, -1, "", "CPU_", 1}}, false, 2},

    ConfigParams{"CPU(3),GPU.1,OTHER",
                 {{"CPU", {}, 3, "", "CPU_", 0},
                  {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1},
                  {"OTHER", {}, -1, "", "OTHER_", 2}},
                 false,
                 3},
    ConfigParams{"OTHER,GPU.1,CPU(3)",
                 {{"OTHER", {}, -1, "", "OTHER_", 0},
                  {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1},
                  {"CPU", {}, 3, "", "CPU_", 2}},
                 false,
                 3}};

const std::vector<ConfigParams> testConfigsNoID = {
    ConfigParams{"CPU,GPU,OTHER",
                 {{"CPU", {}, -1, "", "CPU_", 0},
                  {"GPU", {}, -1, "0", std::string(igpuFullDeviceName) + "_0", 1},
                  {"OTHER", {}, -1, "", "OTHER_", 2}},
                 false,
                 3},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         ParseMetaDeviceTest,
                         ::testing::ValuesIn(testConfigs),
                         ParseMetaDeviceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         ParseMetaDeviceNoIDTest,
                         ::testing::ValuesIn(testConfigsNoID),
                         ParseMetaDeviceTest::getTestCaseName);

// toDo need add test for ParseMetaDevices(_, config) to check device config of
// return metaDevices

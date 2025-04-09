// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
#include "openvino/runtime/internal_properties.hpp"

using Config = std::map<std::string, std::string>;
using namespace ov::mock_auto_plugin;

const std::vector<std::string> availableDevs = {"CPU", "GPU", "NPU"};
const std::vector<std::string> availableDevsWithId = {"CPU", "GPU.0", "GPU.1", "NPU"};
const std::vector<std::string> netPrecisions = {"FP32", "FP16", "INT8", "BIN"};
const char npuUuid[] = "000102030405060708090a0b0c0d0e0f";
using Params = std::tuple<std::string, std::string, int>;
using ConfigParams = std::tuple<std::vector<std::string>,  // Available devices retrieved from Core
                                Params                     // Params {devicePriority, expect metaDevices}
                                >;
class GetDeviceListTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        Params priorityAndMetaDev;
        std::string priorityDevices;
        std::string metaDevices;
        std::vector<std::string> availableDevices;
        int expectedTimes = 0;
        std::tie(availableDevices, priorityAndMetaDev) = obj.param;
        std::tie(priorityDevices, metaDevices, expectedTimes) = priorityAndMetaDev;
        std::ostringstream result;
        result << "priorityDevices_" << priorityDevices;
        result << "_expectedDevices_" << metaDevices;
        result << "_expectedCallAvailableTimes_" << expectedTimes;
        result << "_availableDevicesList";
        std::string devicesStr;
        for (auto&& device : availableDevices) {
            devicesStr += "_" + device;
        }
        result << devicesStr;
        return result.str();
    }

    void SetUp() override {
        ON_CALL(*plugin, get_device_list).WillByDefault([this](const ov::AnyMap& config) {
            return plugin->Plugin::get_device_list(config);
        });
    }
};

TEST_P(GetDeviceListTest, GetDeviceListTestWithExcludeList) {
    // get Parameter
    Params priorityAndMetaDev;
    std::string priorityDevices;
    std::string metaDevices;
    std::vector<std::string> availableDevs;
    int expectedTimes = 0;
    std::tie(availableDevs, priorityAndMetaDev) = this->GetParam();
    std::tie(priorityDevices, metaDevices, expectedTimes) = priorityAndMetaDev;
    std::vector<std::string> deviceIDs = {"0", "1"};
    if (availableDevs != availableDevsWithId) {
        deviceIDs.clear();
        if (priorityDevices.find("GPU.0") != std::string::npos)
            deviceIDs.push_back("0");
    }
    ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::available_devices.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(deviceIDs));
    ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));

    EXPECT_CALL(*core, get_available_devices()).Times(expectedTimes);
    if (metaDevices == "") {
        EXPECT_THROW(plugin->get_device_list({ov::device::priorities(priorityDevices)}), ov::Exception);
    } else {
        std::string result;
        OV_ASSERT_NO_THROW(result = plugin->get_device_list({ov::device::priorities(priorityDevices)}));
        EXPECT_EQ(result, metaDevices);
    }
}

using GetDeviceListTestWithNotInteldGPU = GetDeviceListTest;
TEST_P(GetDeviceListTestWithNotInteldGPU, GetDeviceListTestWithExcludeList) {
    // get Parameter
    Params priorityAndMetaDev;
    std::string priorityDevices;
    std::string metaDevices;
    std::vector<std::string> availableDevs;
    int expectedTimes = 0;
    std::tie(availableDevs, priorityAndMetaDev) = this->GetParam();
    std::tie(priorityDevices, metaDevices, expectedTimes) = priorityAndMetaDev;
    std::vector<std::string> deviceIDs = {"0", "1"};
    if (availableDevs != availableDevsWithId) {
        deviceIDs.clear();
        if (priorityDevices.find("GPU.0") != std::string::npos)
            deviceIDs.push_back("0");
    }
    ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::available_devices.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(deviceIDs));
    ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));
    std::string dgpuArchitecture = "GPU: vendor=0x10DE arch=0";
    ON_CALL(*core, get_property(StrEq("GPU.1"), StrEq(ov::device::architecture.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(dgpuArchitecture));
    EXPECT_CALL(*core, get_available_devices()).Times(expectedTimes);
    if (metaDevices == "") {
        EXPECT_THROW(plugin->get_device_list({ov::device::priorities(priorityDevices)}), ov::Exception);
    } else {
        std::string result;
        OV_ASSERT_NO_THROW(result = plugin->get_device_list({ov::device::priorities(priorityDevices)}));
        EXPECT_EQ(result, metaDevices);
    }
}

const std::vector<Params> testConfigsWithId = {
    Params{" ", " ", 0},
    Params{"", "CPU,GPU.0,GPU.1", 1},
    Params{"CPU, ", "CPU, ", 0},
    Params{" ,CPU", " ,CPU", 0},
    Params{"CPU,", "CPU", 0},
    Params{"CPU,,GPU", "CPU,GPU.0,GPU.1", 0},
    Params{"CPU, ,GPU", "CPU, ,GPU.0,GPU.1", 0},
    Params{"CPU,GPU,GPU.1", "CPU,GPU.0,GPU.1", 0},
    Params{"CPU,GPU,NPU,INVALID_DEVICE", "CPU,GPU.0,GPU.1,NPU,INVALID_DEVICE", 0},
    Params{"NPU,GPU,CPU,-GPU.0", "NPU,GPU.1,CPU", 0},
    Params{"-GPU.0,GPU,CPU", "GPU.1,CPU", 0},
    Params{"-GPU.0,GPU", "GPU.1", 0},
    Params{"-GPU,GPU.0", "GPU.0", 0},
    Params{"-GPU.0", "CPU,GPU.1", 1},
    Params{"-GPU.0,-GPU.1", "CPU", 1},
    Params{"-GPU.0,-GPU.1,INVALID_DEVICE", "INVALID_DEVICE", 0},
    Params{"-GPU.0,-GPU.1,-INVALID_DEVICE", "CPU", 1},
    Params{"-GPU.0,-GPU.1,-CPU", "", 1},
    Params{"GPU,-GPU.0", "GPU.1", 0},
    Params{"-GPU,CPU", "CPU", 0},
    Params{"-GPU,-CPU", "", 1},
    Params{"GPU.0,-GPU", "GPU.0", 0},
    Params{"-GPU.0,-CPU", "GPU.1", 1}};

const std::vector<Params> testConfigs = {Params{" ", " ", 0},
                                         Params{"", "CPU,GPU", 1},
                                         Params{"GPU", "GPU", 0},
                                         Params{"GPU.0", "GPU.0", 0},
                                         Params{"GPU,GPU.0", "GPU.0", 0},
                                         Params{"CPU", "CPU", 0},
                                         Params{" ,CPU", " ,CPU", 0},
                                         Params{" ,GPU", " ,GPU", 0},
                                         Params{"GPU, ", "GPU, ", 0},
                                         Params{"CPU,GPU", "CPU,GPU", 0},
                                         Params{"CPU,-GPU", "CPU", 0},
                                         Params{"CPU,-GPU,GPU.0", "CPU,GPU.0", 0},
                                         Params{"CPU,GPU,-GPU.0", "CPU", 0},
                                         Params{"CPU,GPU,-GPU.1", "CPU,GPU", 0},
                                         Params{"CPU,GPU.0,GPU", "CPU,GPU.0", 0},
                                         Params{"CPU,GPU,GPU.0", "CPU,GPU.0", 0},
                                         Params{"CPU,GPU,GPU.1", "CPU,GPU,GPU.1", 0},
                                         Params{"CPU,GPU.1,GPU", "CPU,GPU.1,GPU", 0},
                                         Params{"CPU,NPU", "CPU,NPU", 0},
                                         Params{"CPU,-NPU", "CPU", 0},
                                         Params{"INVALID_DEVICE", "INVALID_DEVICE", 0},
                                         Params{"CPU,-INVALID_DEVICE", "CPU", 0},
                                         Params{"CPU,INVALID_DEVICE", "CPU,INVALID_DEVICE", 0},
                                         Params{"-CPU,INVALID_DEVICE", "INVALID_DEVICE", 0},
                                         Params{"CPU,GPU,NPU", "CPU,GPU,NPU", 0}};

const std::vector<Params> testConfigsWithIdNotInteldGPU = {
    Params{" ", " ", 0},
    Params{"", "CPU,GPU.0", 1},
    Params{"CPU, ", "CPU, ", 0},
    Params{" ,CPU", " ,CPU", 0},
    Params{"CPU,", "CPU", 0},
    Params{"CPU,,GPU", "CPU,GPU.0,GPU.1", 0},
    Params{"CPU, ,GPU", "CPU, ,GPU.0,GPU.1", 0},
    Params{"CPU,GPU,GPU.1", "CPU,GPU.0,GPU.1", 0},
    Params{"CPU,GPU,NPU,INVALID_DEVICE", "CPU,GPU.0,GPU.1,NPU,INVALID_DEVICE", 0},
    Params{"NPU,GPU,CPU,-GPU.0", "NPU,GPU.1,CPU", 0},
    Params{"-GPU.0,GPU,CPU", "GPU.1,CPU", 0},
    Params{"-GPU.0,GPU", "GPU.1", 0},
    Params{"-GPU,GPU.0", "GPU.0", 0},
    Params{"-GPU.0", "CPU", 1},
    Params{"-GPU.0,-GPU.1", "CPU", 1},
    Params{"-GPU.0,-GPU.1,INVALID_DEVICE", "INVALID_DEVICE", 0},
    Params{"-GPU.0,-GPU.1,-INVALID_DEVICE", "CPU", 1},
    Params{"-GPU.0,-GPU.1,-CPU", "", 1},
    Params{"GPU,-GPU.0", "GPU.1", 0},
    Params{"GPU.0,-GPU", "GPU.0", 0},
    Params{"GPU", "GPU.0,GPU.1", 0},
    Params{"GPU.0", "GPU.0", 0},
    Params{"GPU.1", "GPU.1", 0},
    Params{"-CPU", "GPU.0", 1},
    Params{"-CPU,-GPU", "", 1},
    Params{"-CPU,-GPU.0", "", 1},
    Params{"-CPU,-GPU.1", "GPU.0", 1},
    Params{"-GPU,CPU", "CPU", 0},
    Params{"-GPU.0,-CPU", "", 1}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_GetDeviceListWithID,
                         GetDeviceListTest,
                         ::testing::Combine(::testing::Values(availableDevsWithId),
                                            ::testing::ValuesIn(testConfigsWithId)),
                         GetDeviceListTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_GetDeviceList,
                         GetDeviceListTest,
                         ::testing::Combine(::testing::Values(availableDevs), ::testing::ValuesIn(testConfigs)),
                         GetDeviceListTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_GetDeviceListNotInteldGPU,
                         GetDeviceListTestWithNotInteldGPU,
                         ::testing::Combine(::testing::Values(availableDevsWithId),
                                            ::testing::ValuesIn(testConfigsWithIdNotInteldGPU)),
                         GetDeviceListTestWithNotInteldGPU::getTestCaseName);

// toDo need add test for ParseMetaDevices(_, config) to check device config of
// return metaDevices

using ConfigFilterParams = std::tuple<std::map<std::string, double>,                    // utilization threshold,
                                      std::vector<ov::auto_plugin::DeviceInformation>,  // device candidate list
                                      std::map<std::string, double>,                    // device utilization
                                      std::list<ov::auto_plugin::DeviceInformation>     // expected device list
                                      >;
class GetValidDeviceListTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigFilterParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigFilterParams> obj) {
        std::map<std::string, double> threshold;
        std::vector<ov::auto_plugin::DeviceInformation> devicesInfo;
        std::list<ov::auto_plugin::DeviceInformation> filteredDevicesInfo;
        std::map<std::string, double> deviceUtilization;
        std::tie(threshold, devicesInfo, deviceUtilization, filteredDevicesInfo) = obj.param;
        std::ostringstream result;
        for (const auto& item : threshold) {
            result << item.first << "_utilizationThreshold_" << item.second << "_";
        }
        result << "candidateDeviceList_";
        for (auto dev : devicesInfo)
            result << dev.device_name << "_priority_" << dev.device_priority << "_";

        result << "deviceUtilization_";
        for (auto item : deviceUtilization) {
            result << item.first << "_" << item.second << "_";
        }

        result << "expectedFilteredDeviceList_";
        for (auto dev : filteredDevicesInfo)
            result << dev.device_name << "_priority_" << dev.device_priority << "_";
        return result.str();
    }

    void SetUp() override {
        std::tie(threshold, devicesInfo, deviceUtilization, filteredDevicesInfo) = GetParam();
        std::vector<std::string> npuCability = {"FP32", "FP16", "INT8", "BIN"};
        ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_NPU), StrEq(ov::device::capabilities.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(npuCability));
        ov::AnyMap config = {};
        ON_CALL(*plugin, get_property(StrEq(ov::intel_auto::devices_utilization_threshold.name()), config))
            .WillByDefault(Return(ov::Any(threshold)));
        ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::device::luid.name()), _))
            .WillByDefault(Return(ov::Any("00000000")));
        ON_CALL(*core, get_property(StrEq("GPU.0"), StrEq(ov::device::luid.name()), _))
            .WillByDefault(Return(ov::Any("00000001")));
        ON_CALL(*core, get_property(StrEq("GPU.1"), StrEq(ov::device::luid.name()), _))
            .WillByDefault(Return(ov::Any("00000002")));
        ON_CALL(*core, get_property(StrEq("NPU"), StrEq(ov::device::luid.name()), _))
            .WillByDefault(Return(ov::Any(npuUuid)));
        ON_CALL(*core,
                get_property(StrEq(ov::test::utils::DEVICE_AUTO),
                             StrEq(ov::intel_auto::devices_utilization_threshold.name()),
                             _))
            .WillByDefault(Return(ov::Any(threshold)));
        ON_CALL(*plugin, get_device_utilization).WillByDefault([this](const std::string& luid) {
            return deviceUtilization;
        });
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  const std::map<std::string, double>& utilization_thresholds) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision, utilization_thresholds);
            });
    }

protected:
    std::map<std::string, double> threshold;
    std::vector<ov::auto_plugin::DeviceInformation> devicesInfo;
    std::list<ov::auto_plugin::DeviceInformation> filteredDevicesInfo;
    std::map<std::string, double> deviceUtilization;
};

TEST_P(GetValidDeviceListTest, GetValidFilteredDeviceListTest) {
    // get Parameter
    std::list<ov::auto_plugin::DeviceInformation> result;
    for (auto& precision : netPrecisions) {
        ASSERT_NO_THROW(result = plugin->get_valid_device(devicesInfo, precision, threshold));
        auto actualSize = result.size();
        auto expectedSize = filteredDevicesInfo.size();
        EXPECT_EQ(actualSize, expectedSize);
        // EXPECT_EQ(result, filteredDevicesInfo);
    }
}
const std::map<std::string, double> testUtilizThreshold_15 = {{"CPU", 15},
                                                              {"GPU", 15},
                                                              {"GPU.0", 15},
                                                              {"GPU.1", 15},
                                                              {"NPU", 15}};
const std::map<std::string, double> testUtilizThreshold_80 = {{"CPU", 80},
                                                              {"GPU", 80},
                                                              {"GPU.0", 80},
                                                              {"GPU.1", 80},
                                                              {"NPU", 80}};
const std::map<std::string, double> testUtilizThreshold_100 = {{"CPU", 100},
                                                               {"GPU", 100},
                                                               {"GPU.0", 100},
                                                               {"GPU.1", 100},
                                                               {"NPU", 100}};
const std::vector<ConfigFilterParams> testValidConfigs = {
    ConfigFilterParams{testUtilizThreshold_80,                 // utilization threshold
                       {{"CPU", {}, -1, "01", "CPU_01", 0}},   // device candidates list
                       {{"Total", 15.3}},                      // device utilization
                       {{"CPU", {}, -1, "01", "CPU_01", 0}}},  // expected list of device candidates after filtering
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}},
                       {{"Total", 85.2}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"Total", 15.3}, {"00000000", 20}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"Total", 15.3}, {npuUuid, 20}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU", 0}}},
    ConfigFilterParams{
        testUtilizThreshold_80,
        {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}, {"NPU", {}, -1, "01", "NPU", 0}},
        {{"Total", 85.2}, {"00000000", 20}, {npuUuid, 20}},
        {{"GPU", {}, -1, "01", "GPU", 0}, {"NPU", {}, -1, "01", "NPU", 0}}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"Total", 85.2}, {"00000000", 20}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"Total", 85.2}, {"00000000", 20}},
                       {{"GPU", {}, -1, "01", "GPU", 2}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"NPU", {}, -1, "01", "NPU", 2}},
                       {{"Total", 85.2}, {npuUuid, 20}},
                       {{"NPU", {}, -1, "01", "NPU", 2}}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"Total", 85.2}, {"00000000", 20}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
                       {{"Total", 85.2}, {"00000001", 20}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.0", {}, -1, "01", "iGPU_01", 0}}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"Total", 85.2}, {"00000001", 20}, {"00000002", 50}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0},
                        {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"Total", 85.2}, {"00000001", 20}, {"00000002", 50}, {npuUuid, 30}},
                       {{"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0},
                        {"NPU", {}, -1, "01", "NPU", 0}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0},
                        {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"Total", 85.2}, {"00000001", 82}, {"00000002", 50}, {npuUuid, 30}},
                       {{"GPU.1", {}, -1, "01", "dGPU_01", 0}, {"NPU", {}, -1, "01", "NPU", 0}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"Total", 15.2}, {"00000001", 90}, {"00000002", 50}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.1", {}, -1, "01", "dGPU_01", 0}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"Total", 15.2}, {"00000001", 10}, {"00000002", 90}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.0", {}, -1, "01", "iGPU_01", 0}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"Total", 15.2}, {"00000001", 10}, {"00000002", 90}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU.0", {}, -1, "01", "iGPU_01", 2}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3},
                        {"NPU", {}, -1, "01", "NPU", 4}},
                       {{"Total", 15.2}, {"00000001", 10}, {"00000002", 90}, {npuUuid, 88}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU.0", {}, -1, "01", "iGPU_01", 2}}},
    ConfigFilterParams{testUtilizThreshold_100,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3},
                        {"NPU", {}, -1, "01", "NPU", 4}},
                       {{"Total", 200}, {"00000001", 200}, {"00000002", 200}, {npuUuid, 200}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3},
                        {"NPU", {}, -1, "01", "NPU", 4}}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"Total", 15.2}, {"00000001", 90}, {"00000002", 10}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU.1", {}, -1, "01", "dGPU_01", 3}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_GetValidDeviceList,
                         GetValidDeviceListTest,
                         ::testing::ValuesIn(testValidConfigs),
                         GetValidDeviceListTest::getTestCaseName);
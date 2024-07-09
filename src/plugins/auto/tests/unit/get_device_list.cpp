// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"

using Config = std::map<std::string, std::string>;
using namespace ov::mock_auto_plugin;

const std::vector<std::string> availableDevs = {"CPU", "GPU", "NPU"};
const std::vector<std::string> availableDevsWithId = {"CPU", "GPU.0", "GPU.1", "NPU"};
const std::vector<std::string> netPrecisions = {"FP32", "FP16", "INT8", "BIN"};
using Params = std::tuple<std::string, std::string>;
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
        std::tie(availableDevices, priorityAndMetaDev) = obj.param;
        std::tie(priorityDevices, metaDevices) = priorityAndMetaDev;
        std::ostringstream result;
        result << "priorityDevices_" << priorityDevices;
        result << "_expectedDevices" << metaDevices;
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
    std::tie(availableDevs, priorityAndMetaDev) = this->GetParam();
    std::tie(priorityDevices, metaDevices) = priorityAndMetaDev;

    ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));

    EXPECT_CALL(*core, get_available_devices()).Times(1);
    if (metaDevices == "") {
        EXPECT_THROW(plugin->get_device_list({ov::device::priorities(priorityDevices)}), ov::Exception);
    } else {
        std::string result;
        ASSERT_NO_THROW(result = plugin->get_device_list({ov::device::priorities(priorityDevices)}));
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
    std::tie(availableDevs, priorityAndMetaDev) = this->GetParam();
    std::tie(priorityDevices, metaDevices) = priorityAndMetaDev;

    ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));
    std::string dgpuArchitecture = "GPU: vendor=0x10DE arch=0";
    ON_CALL(*core, get_property(StrEq("GPU.1"), StrEq(ov::device::architecture.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(dgpuArchitecture));
    EXPECT_CALL(*core, get_available_devices()).Times(1);
    if (metaDevices == "") {
        EXPECT_THROW(plugin->get_device_list({ov::device::priorities(priorityDevices)}), ov::Exception);
    } else {
        std::string result;
        ASSERT_NO_THROW(result = plugin->get_device_list({ov::device::priorities(priorityDevices)}));
        EXPECT_EQ(result, metaDevices);
    }
}

const std::vector<Params> testConfigsWithId = {
    Params{" ", " "},
    Params{"", "CPU,GPU.0,GPU.1"},
    Params{"CPU, ", "CPU, "},
    Params{" ,CPU", " ,CPU"},
    Params{"CPU,", "CPU"},
    Params{"CPU,,GPU", "CPU,GPU.0,GPU.1"},
    Params{"CPU, ,GPU", "CPU, ,GPU.0,GPU.1"},
    Params{"CPU,GPU,GPU.1", "CPU,GPU.0,GPU.1"},
    Params{"CPU,GPU,NPU,INVALID_DEVICE", "CPU,GPU.0,GPU.1,NPU,INVALID_DEVICE"},
    Params{"NPU,GPU,CPU,-GPU.0", "NPU,GPU.1,CPU"},
    Params{"-GPU.0,GPU,CPU", "GPU.1,CPU"},
    Params{"-GPU.0,GPU", "GPU.1"},
    Params{"-GPU,GPU.0", "GPU.0"},
    Params{"-GPU.0", "CPU,GPU.1"},
    Params{"-GPU.0,-GPU.1", "CPU"},
    Params{"-GPU.0,-GPU.1,INVALID_DEVICE", "INVALID_DEVICE"},
    Params{"-GPU.0,-GPU.1,-INVALID_DEVICE", "CPU"},
    Params{"-GPU.0,-GPU.1,-CPU", ""},
    Params{"GPU,-GPU.0", "GPU.1"},
    Params{"-GPU,CPU", "CPU"},
    Params{"-GPU,-CPU", ""},
    Params{"GPU.0,-GPU", "GPU.0"},
    Params{"-GPU.0,-CPU", "GPU.1"}};

const std::vector<Params> testConfigs = {Params{" ", " "},
                                         Params{"", "CPU,GPU"},
                                         Params{"GPU", "GPU"},
                                         Params{"GPU.0", "GPU.0"},
                                         Params{"GPU,GPU.0", "GPU"},
                                         Params{"CPU", "CPU"},
                                         Params{" ,CPU", " ,CPU"},
                                         Params{" ,GPU", " ,GPU"},
                                         Params{"GPU, ", "GPU, "},
                                         Params{"CPU,GPU", "CPU,GPU"},
                                         Params{"CPU,-GPU", "CPU"},
                                         Params{"CPU,-GPU,GPU.0", "CPU,GPU.0"},
                                         Params{"CPU,-GPU,GPU.1", "CPU,GPU.1"},
                                         Params{"CPU,GPU,-GPU.0", "CPU"},
                                         Params{"CPU,GPU,-GPU.1", "CPU,GPU"},
                                         Params{"CPU,GPU.0,GPU", "CPU,GPU"},
                                         Params{"CPU,GPU,GPU.0", "CPU,GPU"},
                                         Params{"CPU,GPU,GPU.1", "CPU,GPU,GPU.1"},
                                         Params{"CPU,GPU.1,GPU", "CPU,GPU.1,GPU"},
                                         Params{"CPU,NPU", "CPU,NPU"},
                                         Params{"CPU,-NPU", "CPU"},
                                         Params{"INVALID_DEVICE", "INVALID_DEVICE"},
                                         Params{"CPU,-INVALID_DEVICE", "CPU"},
                                         Params{"CPU,INVALID_DEVICE", "CPU,INVALID_DEVICE"},
                                         Params{"-CPU,INVALID_DEVICE", "INVALID_DEVICE"},
                                         Params{"CPU,GPU,NPU", "CPU,GPU,NPU"}};

const std::vector<Params> testConfigsWithIdNotInteldGPU = {
    Params{" ", " "},
    Params{"", "CPU,GPU.0"},
    Params{"CPU, ", "CPU, "},
    Params{" ,CPU", " ,CPU"},
    Params{"CPU,", "CPU"},
    Params{"CPU,,GPU", "CPU,GPU.0,GPU.1"},
    Params{"CPU, ,GPU", "CPU, ,GPU.0,GPU.1"},
    Params{"CPU,GPU,GPU.1", "CPU,GPU.0,GPU.1"},
    Params{"CPU,GPU,NPU,INVALID_DEVICE", "CPU,GPU.0,GPU.1,NPU,INVALID_DEVICE"},
    Params{"NPU,GPU,CPU,-GPU.0", "NPU,GPU.1,CPU"},
    Params{"-GPU.0,GPU,CPU", "GPU.1,CPU"},
    Params{"-GPU.0,GPU", "GPU.1"},
    Params{"-GPU,GPU.0", "GPU.0"},
    Params{"-GPU.0", "CPU"},
    Params{"-GPU.0,-GPU.1", "CPU"},
    Params{"-GPU.0,-GPU.1,INVALID_DEVICE", "INVALID_DEVICE"},
    Params{"-GPU.0,-GPU.1,-INVALID_DEVICE", "CPU"},
    Params{"-GPU.0,-GPU.1,-CPU", ""},
    Params{"GPU,-GPU.0", "GPU.1"},
    Params{"GPU.0,-GPU", "GPU.0"},
    Params{"GPU", "GPU.0,GPU.1"},
    Params{"GPU.0", "GPU.0"},
    Params{"GPU.1", "GPU.1"},
    Params{"-CPU", "GPU.0"},
    Params{"-CPU,-GPU", ""},
    Params{"-CPU,-GPU.0", ""},
    Params{"-CPU,-GPU.1", "GPU.0"},
    Params{"-GPU,CPU", "CPU"},
    Params{"-GPU.0,-CPU", ""}};

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

using ConfigFilterParams = std::tuple<double,                                                // utilization threshold,
                                      std::vector<ov::auto_plugin::DeviceInformation>,       // device candidate list
                                      std::map<std::string, std::map<std::string, double>>,  // device utilization
                                      std::list<ov::auto_plugin::DeviceInformation>          // expected device list
                                      >;
class GetValidDeviceListTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigFilterParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigFilterParams> obj) {
        double threshold;
        std::vector<ov::auto_plugin::DeviceInformation> devicesInfo;
        std::list<ov::auto_plugin::DeviceInformation> filteredDevicesInfo;
        std::map<std::string, std::map<std::string, double>> deviceUtilization;
        std::tie(threshold, devicesInfo, deviceUtilization, filteredDevicesInfo) = obj.param;
        std::ostringstream result;
        result << "utilizationThreshold_" << threshold << "_";
        result << "candidateDeviceList_";
        for (auto dev : devicesInfo)
            result << dev.device_name << "_priority_" << dev.device_priority << "_";

        result << "deviceUtilization_";
        for (auto dev : deviceUtilization) {
            result << dev.first << "_";
            for (auto& item : dev.second)
                result << item.first << "_" << item.second << "_";
        }

        result << "expectedFilteredDeviceList_";
        for (auto dev : filteredDevicesInfo)
            result << dev.device_name << "_priority_" << dev.device_priority << "_";
        return result.str();
    }

    void SetUp() override {
        std::tie(threshold, devicesInfo, deviceUtilization, filteredDevicesInfo) = GetParam();
        ov::AnyMap config = {};
        ON_CALL(*plugin, get_property(StrEq(ov::intel_auto::device_utilization_threshold.name()), config))
            .WillByDefault(Return(ov::Any(threshold)));
        ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::device::luid.name()), _))
            .WillByDefault(Return(ov::Any("00000000")));
        ON_CALL(*core, get_property(StrEq("GPU.0"), StrEq(ov::device::luid.name()), _))
            .WillByDefault(Return(ov::Any("00000001")));
        ON_CALL(*core, get_property(StrEq("GPU.1"), StrEq(ov::device::luid.name()), _))
            .WillByDefault(Return(ov::Any("00000002")));
        ON_CALL(*core,
                get_property(StrEq(ov::test::utils::DEVICE_AUTO),
                             StrEq(ov::intel_auto::device_utilization_threshold.name()),
                             _))
            .WillByDefault(Return(ov::Any(threshold)));
        ON_CALL(*plugin, get_device_utilization).WillByDefault([this](const std::string& device) {
            ov::DeviceIDParser parsed{device};
            return deviceUtilization[parsed.get_device_name()];
        });
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision);
            });
    }

protected:
    double threshold;
    std::vector<ov::auto_plugin::DeviceInformation> devicesInfo;
    std::list<ov::auto_plugin::DeviceInformation> filteredDevicesInfo;
    std::map<std::string, std::map<std::string, double>> deviceUtilization;
};

TEST_P(GetValidDeviceListTest, GetValidFilteredDeviceListTest) {
    // get Parameter
    std::list<ov::auto_plugin::DeviceInformation> result;
    for (auto& precision : netPrecisions) {
        ASSERT_NO_THROW(result = plugin->get_valid_device(devicesInfo, precision));
        int actualSize = result.size();
        int expectedSize = filteredDevicesInfo.size();
        EXPECT_EQ(actualSize, expectedSize);
        // EXPECT_EQ(result, filteredDevicesInfo);
    }
}

const std::vector<ConfigFilterParams> testValidConfigs = {
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}},
                       {{"CPU", {{"Total", 15.3}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}},
                       {{"CPU", {{"Total", 85.2}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"CPU", {{"Total", 15.3}}}, {"GPU", {{"00000000", 20}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"CPU", {{"Total", 85.2}}}, {"GPU", {{"00000000", 20}}}},
                       {{"GPU", {}, -1, "01", "GPU", 0}}},
    ConfigFilterParams{15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"CPU", {{"Total", 85.2}}}, {"GPU", {{"00000000", 20}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"CPU", {{"Total", 85.2}}}, {"GPU", {{"00000000", 20}}}},
                       {{"GPU", {}, -1, "01", "GPU", 2}}},
    ConfigFilterParams{15,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"CPU", {{"Total", 85.2}}}, {"GPU", {{"00000000", 20}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}}},
    ConfigFilterParams{15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
                       {{"CPU", {{"Total", 85.2}}}, {"GPU", {{"00000001", 20}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.0", {}, -1, "01", "iGPU_01", 0}}},
    ConfigFilterParams{15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"CPU", {{"Total", 85.2}}}, {"GPU", {{"00000001", 20}, {"00000002", 50}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"CPU", {{"Total", 85.2}}}, {"GPU", {{"00000001", 20}, {"00000002", 50}}}},
                       {{"GPU.0", {}, -1, "01", "iGPU_01", 0}, {"GPU.1", {}, -1, "01", "dGPU_01", 0}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"CPU", {{"Total", 15.2}}}, {"GPU", {{"00000001", 90}, {"00000002", 50}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.1", {}, -1, "01", "dGPU_01", 0}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"CPU", {{"Total", 15.2}}}, {"GPU", {{"00000001", 10}, {"00000002", 90}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.0", {}, -1, "01", "iGPU_01", 0}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"CPU", {{"Total", 15.2}}}, {"GPU", {{"00000001", 10}, {"00000002", 90}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU.0", {}, -1, "01", "iGPU_01", 2}}},
    ConfigFilterParams{100,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"CPU", {{"Total", 200}}}, {"GPU", {{"00000001", 200}, {"00000002", 200}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}}},
    ConfigFilterParams{80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"CPU", {{"Total", 15.2}}}, {"GPU", {{"00000001", 90}, {"00000002", 10}}}},
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU.1", {}, -1, "01", "dGPU_01", 3}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_GetValidDeviceList,
                         GetValidDeviceListTest,
                         ::testing::ValuesIn(testValidConfigs),
                         GetValidDeviceListTest::getTestCaseName);
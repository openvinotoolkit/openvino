// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"

using namespace ov::mock_auto_plugin;
using ConfigParams = std::tuple<std::string,                     // netPrecision
                                std::vector<DeviceInformation>,  // metaDevices for select
                                DeviceInformation,               // expect DeviceInformation
                                bool,                            // throw exception
                                bool,                            // enabledevice_priority
                                bool                             // reverse total device
                                >;

const DeviceInformation CPU_INFO = {ov::test::utils::DEVICE_CPU, {}, 2, "01", "CPU_01"};
const DeviceInformation IGPU_INFO = {"GPU.0", {}, 2, "01", "iGPU_01"};
const DeviceInformation DGPU_INFO = {"GPU.1", {}, 2, "01", "dGPU_01"};
const DeviceInformation OTHERS_INFO = {"OTHERS", {}, 2, "01", "OTHERS"};
const char npuUuid[] = "000102030405060708090a0b0c0d0e0f";
const std::vector<DeviceInformation> fp32DeviceVector = {DGPU_INFO, IGPU_INFO, OTHERS_INFO, CPU_INFO};
const std::vector<DeviceInformation> fp16DeviceVector = {DGPU_INFO, IGPU_INFO, OTHERS_INFO, CPU_INFO};
const std::vector<DeviceInformation> int8DeviceVector = {DGPU_INFO, IGPU_INFO, CPU_INFO};
const std::vector<DeviceInformation> binDeviceVector = {DGPU_INFO, IGPU_INFO, CPU_INFO};
std::map<std::string, const std::vector<DeviceInformation>> devicesMap = {{"FP32", fp32DeviceVector},
                                                                          {"FP16", fp16DeviceVector},
                                                                          {"INT8", int8DeviceVector},
                                                                          {"BIN", binDeviceVector}};
const std::vector<DeviceInformation> totalDevices = {DGPU_INFO, IGPU_INFO, OTHERS_INFO, CPU_INFO};
const std::vector<DeviceInformation> reverseTotalDevices = {CPU_INFO, OTHERS_INFO, IGPU_INFO, DGPU_INFO};
const std::vector<std::string> netPrecisions = {"FP32", "FP16", "INT8", "BIN"};
std::vector<ConfigParams> testConfigs;

class SelectDeviceTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string netPrecision;
        std::vector<DeviceInformation> devices;
        DeviceInformation expect;
        bool throwExcept;
        bool enabledevice_priority;
        bool reverse;
        std::tie(netPrecision, devices, expect, throwExcept, enabledevice_priority, reverse) = obj.param;
        std::ostringstream result;
        result << "_netPrecision_" << netPrecision;
        for (auto& item : devices) {
            result << "_device_" << item.unique_name;
        }
        result << "_expect_" << expect.unique_name;
        if (throwExcept) {
            result << "_throwExcept_true";
        } else {
            result << "_throwExcept_false";
        }

        if (enabledevice_priority) {
            result << "_enabledevice_priority_true";
        } else {
            result << "_enabledevice_priority_false";
        }

        if (reverse) {
            result << "_reverseTotalDevice_true";
        } else {
            result << "_reverseTotalDevice_false";
        }

        return result.str();
    }
    // combine select_num devices from devices and make them to ConfigParams
    // insert the ConfigParams into testConfigs
    static void combine_device(const std::vector<DeviceInformation>& devices,
                               size_t start,
                               size_t* result,
                               size_t result_index,
                               const size_t select_num,
                               std::string& netPrecision,
                               bool enabledevice_priority,
                               bool reverse) {
        for (size_t i = start; i < devices.size() + 1 - result_index; i++) {
            result[result_index - 1] = i;
            if (result_index - 1 == 0) {
                std::vector<DeviceInformation> metaDevices = {};
                int device_priority = 0;
                for (int j = static_cast<int>(select_num) - 1; j >= 0; j--) {
                    auto tmpDevInfo = devices[result[j]];
                    if (enabledevice_priority) {
                        tmpDevInfo.device_priority = device_priority;
                        device_priority++;
                    }
                    metaDevices.push_back(tmpDevInfo);
                }
                // Debug the combine_device
                // for (auto& item : metaDevices) {
                //     std::cout << item.unique_name << "_";
                // }
                // std::cout << netPrecision << std::endl;
                auto& devicesInfo = devicesMap[netPrecision];
                bool find = false;
                DeviceInformation expect;
                if (metaDevices.size() > 1) {
                    if (enabledevice_priority) {
                        std::vector<DeviceInformation> validDevices;
                        for (auto& item : devicesInfo) {
                            auto device = std::find_if(metaDevices.begin(),
                                                       metaDevices.end(),
                                                       [&item](const DeviceInformation& d) -> bool {
                                                           return d.unique_name == item.unique_name;
                                                       });
                            if (device != metaDevices.end()) {
                                validDevices.push_back(*device);
                            }
                        }
                        unsigned int currentdevice_priority = 100;
                        for (auto iter = validDevices.begin(); iter != validDevices.end(); iter++) {
                            if (iter->device_priority < currentdevice_priority) {
                                expect = *iter;
                                currentdevice_priority = iter->device_priority;
                            }
                        }
                        if (currentdevice_priority != 100) {
                            find = true;
                        }
                    } else {
                        for (auto& item : devicesInfo) {
                            auto device = std::find_if(metaDevices.begin(),
                                                       metaDevices.end(),
                                                       [&item](const DeviceInformation& d) -> bool {
                                                           return d.unique_name == item.unique_name;
                                                       });
                            if (device != metaDevices.end()) {
                                find = true;
                                expect = item;
                                break;
                            }
                        }
                    }
                } else if (metaDevices.size() == 1) {
                    find = true;
                    expect = metaDevices[0];
                } else {
                    find = false;
                }
                testConfigs.push_back(
                    std::make_tuple(netPrecision, metaDevices, expect, !find, enabledevice_priority, reverse));
            } else {
                combine_device(devices,
                               i + 1,
                               result,
                               result_index - 1,
                               select_num,
                               netPrecision,
                               enabledevice_priority,
                               reverse);
            }
        }
    }

    static std::vector<ConfigParams> CreateConfigs() {
        auto result = new size_t[totalDevices.size()];
        // test all netPrecision with all possible combine devices
        // netPrecision number is 5
        // device number is 5
        // combine devices is 5!/5! + 5!/(4!*1!) + 5!/(3!*2!) + 5!/(2!*3!) + 5(1!*4!) = 31
        // null device 1
        // total test config num is 32*5 = 160
        for (auto netPrecision : netPrecisions) {
            for (size_t i = 1; i <= totalDevices.size(); i++) {
                combine_device(totalDevices, 0, result, i, i, netPrecision, false, false);
            }
            // test null device
            testConfigs.push_back(ConfigParams{netPrecision, {}, {}, true, false, false});
        }
        // reverse totalDevices for test
        for (auto netPrecision : netPrecisions) {
            for (size_t i = 1; i <= reverseTotalDevices.size(); i++) {
                combine_device(reverseTotalDevices, 0, result, i, i, netPrecision, false, true);
            }
        }

        // add test for enabledevice_priority
        // test case num is 31*5 = 155
        for (auto netPrecision : netPrecisions) {
            for (size_t i = 1; i <= totalDevices.size(); i++) {
                combine_device(totalDevices, 0, result, i, i, netPrecision, true, false);
            }
        }

        // reverse totalDevices for test
        for (auto netPrecision : netPrecisions) {
            for (size_t i = 1; i <= reverseTotalDevices.size(); i++) {
                combine_device(reverseTotalDevices, 0, result, i, i, netPrecision, true, true);
            }
        }
        delete[] result;
        return testConfigs;
    }

    void compare(DeviceInformation& a, DeviceInformation& b) {
        EXPECT_EQ(a.device_name, b.device_name);
        EXPECT_EQ(a.unique_name, b.unique_name);
        EXPECT_EQ(a.default_device_id, b.default_device_id);
    }

    void SetUp() override {
        ON_CALL(*plugin, select_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int priority,
                                  const std::map<std::string, double>& utilization_thresholds) {
                return plugin->Plugin::select_device(metaDevices, netPrecision, priority, utilization_thresholds);
            });
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision);
            });
    }
};

TEST_P(SelectDeviceTest, SelectDevice) {
    // get Parameter
    std::string netPrecision;
    std::vector<DeviceInformation> devices;
    DeviceInformation expect;
    bool throwExcept;
    bool enabledevice_priority;
    bool reverse;
    std::tie(netPrecision, devices, expect, throwExcept, enabledevice_priority, reverse) = this->GetParam();

    EXPECT_CALL(*plugin, select_device(_, _, _, _)).Times(1);
    if (devices.size() >= 1) {
        EXPECT_CALL(*core, get_property(_, _, _)).Times(AtLeast(static_cast<int>(devices.size()) - 1));
    } else {
        EXPECT_CALL(*core, get_property(_, _, _)).Times(0);
    }

    if (throwExcept) {
        ASSERT_THROW(plugin->select_device(devices, netPrecision, 0, {}), ov::Exception);
    } else {
        auto result = plugin->select_device(devices, netPrecision, 0, {});
        compare(result, expect);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SelectDeviceTest,
                         ::testing::ValuesIn(SelectDeviceTest::CreateConfigs()),
                         SelectDeviceTest::getTestCaseName);

using ConfigFilterParams = std::tuple<std::map<std::string, double>,                    // utilization threshold,
                                      std::vector<ov::auto_plugin::DeviceInformation>,  // device candidate list
                                      std::map<std::string, double>,                    // device utilization
                                      ov::auto_plugin::DeviceInformation                // expected selected device
                                      >;
class SelectDeviceWithUtilizationTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigFilterParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigFilterParams> obj) {
        std::map<std::string, double> threshold;
        std::vector<ov::auto_plugin::DeviceInformation> devices;
        ov::auto_plugin::DeviceInformation selectedDeviceInfo;
        std::map<std::string, double> deviceUtilization;
        std::tie(threshold, devices, deviceUtilization, selectedDeviceInfo) = obj.param;
        std::ostringstream result;
        for (const auto& item : threshold) {
            result << item.first << "_utilizationThreshold_" << item.second << "_";
        }
        result << "candidateDeviceList_";
        for (auto dev : devices)
            result << dev.device_name << "_priority_" << dev.device_priority << "_";

        result << "deviceUtilization_";
        for (auto item : deviceUtilization) {
            result << item.first << "_" << item.second << "_";
        }

        result << "expectedSelectedDevice_";
        result << selectedDeviceInfo.device_name << "_priority_" << selectedDeviceInfo.device_priority << "_";
        return result.str();
    }

    void compare(DeviceInformation& a, DeviceInformation& b) {
        EXPECT_EQ(a.device_name, b.device_name);
        EXPECT_EQ(a.unique_name, b.unique_name);
        EXPECT_EQ(a.default_device_id, b.default_device_id);
    }

    void SetUp() override {
        std::tie(threshold, devices, deviceUtilization, selectedDeviceInfo) = GetParam();
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
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision);
            });
    }

protected:
    std::map<std::string, double> threshold;
    std::vector<ov::auto_plugin::DeviceInformation> devices;
    ov::auto_plugin::DeviceInformation selectedDeviceInfo;
    std::map<std::string, double> deviceUtilization;
};

TEST_P(SelectDeviceWithUtilizationTest, selectDeviceWithUtilization) {
    // get Parameter
    std::string netPrecision = "FP32";
    auto result = plugin->select_device(devices, netPrecision, 0, threshold);
    compare(result, selectedDeviceInfo);
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
    ConfigFilterParams{testUtilizThreshold_80,                // utilization threshold
                       {{"CPU", {}, -1, "01", "CPU_01", 0}},  // device candidates list
                       {{"Total", 15.3}},                     // device utilization
                       {"CPU", {}, -1, "01", "CPU_01", 0}},   // expected list of device candidates after filtering
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}},
                       {{"Total", 85.2}},
                       {"CPU", {}, -1, "01", "CPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"Total", 15.3}, {"00000000", 20}},
                       {"GPU", {}, -1, "01", "GPU", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"Total", 15.3}, {npuUuid, 20}},
                       {"NPU", {}, -1, "01", "NPU", 0}},
    ConfigFilterParams{
        testUtilizThreshold_80,
        {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}, {"NPU", {}, -1, "01", "NPU", 0}},
        {{"Total", 85.2}, {"00000000", 20}, {npuUuid, 20}},
        {"GPU", {}, -1, "01", "GPU", 0}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"Total", 85.2}, {"00000000", 20}},
                       {"GPU", {}, -1, "01", "GPU", 0}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"Total", 10}, {"00000000", 20}},
                       {"CPU", {}, -1, "01", "CPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"Total", 25}, {"00000000", 20}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"Total", 90}, {"00000000", 25}},
                       {"GPU", {}, -1, "01", "GPU", 2}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"NPU", {}, -1, "01", "NPU", 2}},
                       {{"Total", 85.2}, {npuUuid, 20}},
                       {"NPU", {}, -1, "01", "NPU", 2}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"Total", 85.2}, {"00000000", 20}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
                       {{"Total", 85.2}, {"00000001", 20}},
                       {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"Total", 85.2}, {"00000001", 20}, {"00000002", 50}},
                       {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0},
                        {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"Total", 85.2}, {"00000001", 20}, {"00000002", 50}, {npuUuid, 30}},
                       {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0},
                        {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"Total", 85.2}, {"00000001", 82}, {"00000002", 50}, {npuUuid, 30}},
                       {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"Total", 15.2}, {"00000001", 90}, {"00000002", 50}},
                       {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"Total", 15.2}, {"00000001", 10}, {"00000002", 90}},
                       {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"Total", 15.2}, {"00000001", 10}, {"00000002", 90}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3},
                        {"NPU", {}, -1, "01", "NPU", 4}},
                       {{"Total", 15.2}, {"00000001", 10}, {"00000002", 90}, {npuUuid, 88}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_100,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3},
                        {"NPU", {}, -1, "01", "NPU", 4}},
                       {{"Total", 200}, {"00000001", 200}, {"00000002", 200}, {npuUuid, 200}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"Total", 15.2}, {"00000001", 90}, {"00000002", 10}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SelectDeviceWithUtilizationTest,
                         ::testing::ValuesIn(testValidConfigs),
                         SelectDeviceWithUtilizationTest::getTestCaseName);
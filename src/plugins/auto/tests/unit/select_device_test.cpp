// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
#include <cctype>

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
    static std::string getTestCaseName(const testing::TestParamInfo<ConfigParams>& obj) {
        const auto& [netPrecision, devices, expect, throwExcept, enabledevice_priority, reverse] = obj.param;
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

    void compare(const DeviceInformation& a, const DeviceInformation& b) {
        EXPECT_EQ(a.device_name, b.device_name);
        EXPECT_EQ(a.unique_name, b.unique_name);
        EXPECT_EQ(a.default_device_id, b.default_device_id);
    }

    void SetUp() override {
        ON_CALL(*plugin, select_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int priority,
                                  const ov::auto_plugin::DeviceSelectionPolicy& selection_policy,
                                  const std::string& low_power_device) {
                return plugin->Plugin::select_device(metaDevices, netPrecision, priority, selection_policy, low_power_device);
            });
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision);
            });
    }
};

TEST_P(SelectDeviceTest, SelectDevice) {
    const auto& [netPrecision, devices, expect, throwExcept, enabledevice_priority, reverse] = this->GetParam();

    EXPECT_CALL(*plugin, select_device(_, _, _, _, _)).Times(1);
    if (devices.size() >= 1) {
        EXPECT_CALL(*core, get_property(_, _, _)).Times(AtLeast(static_cast<int>(devices.size()) - 1));
    } else {
        EXPECT_CALL(*core, get_property(_, _, _)).Times(0);
    }

    if (throwExcept) {
        ASSERT_THROW(plugin->select_device(devices, netPrecision, 0, {}, {}), ov::Exception);
    } else {
        auto result = plugin->select_device(devices, netPrecision, 0, {}, {});
        compare(result, expect);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SelectDeviceTest,
                         ::testing::ValuesIn(SelectDeviceTest::CreateConfigs()),
                         SelectDeviceTest::getTestCaseName);

using ConfigFilterParams = std::tuple<std::unordered_map<std::string, unsigned>,        // utilization threshold,
                                      std::vector<ov::auto_plugin::DeviceInformation>,  // device candidate list
                                      std::map<std::string, float>,                    // device utilization
                                      ov::auto_plugin::DeviceInformation                // expected selected device
                                      >;
class SelectDeviceWithUtilizationTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigFilterParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigFilterParams> obj) {
        std::unordered_map<std::string, unsigned> threshold;
        std::vector<ov::auto_plugin::DeviceInformation> devices;
        ov::auto_plugin::DeviceInformation selectedDeviceInfo;
        std::map<std::string, float> deviceUtilization;
        std::tie(threshold, devices, deviceUtilization, selectedDeviceInfo) = obj.param;
        std::ostringstream result;
        // Sort threshold keys for deterministic test naming
        std::vector<std::string> sorted_keys;
        for (const auto& item : threshold) {
            sorted_keys.push_back(item.first);
        }
        std::sort(sorted_keys.begin(), sorted_keys.end());
        for (const auto& key : sorted_keys) {
            result << key << "_utilizationThreshold_" << threshold.at(key) << "_";
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

        auto sanitize_for_gtest = [](std::string name) {
            for (char& ch : name) {
                const unsigned char c = static_cast<unsigned char>(ch);
                if (!std::isalnum(c) && ch != '_') {
                    ch = '_';
                }
            }
            return name;
        };

        return sanitize_for_gtest(result.str());
    }

    void compare(DeviceInformation& a, DeviceInformation& b) {
        EXPECT_EQ(a.device_name, b.device_name);
        EXPECT_EQ(a.unique_name, b.unique_name);
        EXPECT_EQ(a.default_device_id, b.default_device_id);
    }

    void SetUp() override {
        std::tie(threshold, devices, deviceUtilization, selectedDeviceInfo) = GetParam();
        std::map<std::string, unsigned> properties_thresholds(threshold.begin(), threshold.end());
        std::vector<std::string> npuCapability = {"FP32", "FP16", "INT8", "BIN"};
        ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_NPU), StrEq(ov::device::capabilities.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(npuCapability));
        ov::AnyMap config = {};
        ON_CALL(*plugin, get_property(StrEq(ov::intel_auto::devices_utilization_threshold.name()), config))
            .WillByDefault(Return(ov::Any(properties_thresholds)));
        ON_CALL(*core,
                get_property(StrEq(ov::test::utils::DEVICE_AUTO),
                             StrEq(ov::intel_auto::devices_utilization_threshold.name()),
                             _))
            .WillByDefault(Return(ov::Any(properties_thresholds)));
        ON_CALL(*plugin, get_device_utilization)
            .WillByDefault([this](const std::string& device_name, const std::string& device_type) -> std::optional<float> {
                const auto it = deviceUtilization.find(device_name);
                if (it == deviceUtilization.end()) {
                    return std::nullopt;
                }
                return it->second;
            });
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision);
            });
    }

protected:
    std::unordered_map<std::string, unsigned> threshold;
    std::vector<ov::auto_plugin::DeviceInformation> devices;
    ov::auto_plugin::DeviceInformation selectedDeviceInfo;
    std::map<std::string, float> deviceUtilization;
};

TEST_P(SelectDeviceWithUtilizationTest, selectDeviceWithUtilization) {
    // get Parameter
    std::string netPrecision = "FP32";
    auto result = plugin->select_device(devices, netPrecision, 0, {threshold, {}}, {});
    compare(result, selectedDeviceInfo);
}

const std::unordered_map<std::string, unsigned> testUtilizThreshold_15 = {{"CPU", 15},
                                                                          {"GPU", 15},
                                                                          {"GPU.0", 15},
                                                                          {"GPU.1", 15},
                                                                          {"NPU", 15}};
const std::unordered_map<std::string, unsigned> testUtilizThreshold_80 = {{"CPU", 80},
                                                                          {"GPU", 80},
                                                                          {"GPU.0", 80},
                                                                          {"GPU.1", 80},
                                                                          {"NPU", 80}};
const std::unordered_map<std::string, unsigned> testUtilizThreshold_100 = {{"CPU", 100},
                                                                           {"GPU", 100},
                                                                           {"GPU.0", 100},
                                                                           {"GPU.1", 100},
                                                                           {"NPU", 100}};
const std::vector<ConfigFilterParams> testValidConfigs = {
    ConfigFilterParams{testUtilizThreshold_80,                // utilization threshold
                       {{"CPU", {}, -1, "01", "CPU_01", 0}},  // device candidates list
                       {{"CPU", 15.3f}},                    // device utilization
                       {"CPU", {}, -1, "01", "CPU_01", 0}},   // expected list of device candidates after filtering
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}},
                       {{"CPU", 85.2f}},
                       {"CPU", {}, -1, "01", "CPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"CPU", 15.3f}, {"GPU", 20.5f}},
                       {"GPU", {}, -1, "01", "GPU", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"CPU", 15.3f}, {"NPU", 20.5f}},
                       {"NPU", {}, -1, "01", "NPU", 0}},
    ConfigFilterParams{
        testUtilizThreshold_80,
        {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}, {"NPU", {}, -1, "01", "NPU", 0}},
        {{"CPU", 85.2f}, {"GPU", 20.5f}, {"NPU", 20.5f}},
        {"GPU", {}, -1, "01", "GPU", 0}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"CPU", 85.2f}, {"GPU", 20.5f}},
                       {"GPU", {}, -1, "01", "GPU", 0}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU", {}, -1, "01", "GPU", 0}},
                       {{"CPU", 10.5f}, {"GPU", 20.5f}},
                       {"CPU", {}, -1, "01", "CPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"CPU", 25.5f}, {"GPU", 20.5f}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"CPU", 90.5f}, {"GPU", 25.5f}},
                       {"GPU", {}, -1, "01", "GPU", 2}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"NPU", {}, -1, "01", "NPU", 2}},
                       {{"CPU", 85.5f}, {"NPU", 20.5f}},
                       {"NPU", {}, -1, "01", "NPU", 2}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 1}, {"GPU", {}, -1, "01", "GPU", 2}},
                       {{"CPU", 85.5f}, {"GPU", 20.5f}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0}, {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
                       {{"CPU", 85.5f}, {"GPU.0", 20.5f}},
                       {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_15,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"CPU", 85.5f}, {"GPU.0", 20.5f}, {"GPU.1", 50.5f}},
                       {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0},
                        {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"CPU", 85.5f}, {"GPU.0", 20.5f}, {"GPU.1", 50.5f}, {"NPU", 30.5f}},
                       {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0},
                        {"NPU", {}, -1, "01", "NPU", 0}},
                       {{"CPU", 85.5f}, {"GPU.0", 82.5f}, {"GPU.1", 50.5f}, {"NPU", 30.5f}},
                       {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"CPU", 15.5f}, {"GPU.0", 90.5f}, {"GPU.1", 50.5f}},
                       {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 0},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 0},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                       {{"CPU", 15.5f}, {"GPU.0", 10.5f}, {"GPU.1", 90.5f}},
                       {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"CPU", 15.5f}, {"GPU.0", 10.5f}, {"GPU.1", 90.5f}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3},
                        {"NPU", {}, -1, "01", "NPU", 4}},
                       {{"CPU", 15.5f}, {"GPU.0", 10.5f}, {"GPU.1", 90.5f}, {"NPU", 88.5f}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_100,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3},
                        {"NPU", {}, -1, "01", "NPU", 4}},
                       {{"CPU", 200.0f}, {"GPU.0", 200.0f}, {"GPU.1", 200.0f}, {"NPU", 200.0f}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}},
    ConfigFilterParams{testUtilizThreshold_80,
                       {{"CPU", {}, -1, "01", "CPU_01", 1},
                        {"GPU.0", {}, -1, "01", "iGPU_01", 2},
                        {"GPU.1", {}, -1, "01", "dGPU_01", 3}},
                       {{"CPU", 15.0f}, {"GPU.0", 90.0f}, {"GPU.1", 10.0f}},
                       {"CPU", {}, -1, "01", "CPU_01", 1}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SelectDeviceWithUtilizationTest,
                         ::testing::ValuesIn(testValidConfigs),
                         SelectDeviceWithUtilizationTest::getTestCaseName);


// ------------------------------------------------------------------------------------------
// select_device() end-to-end tests driven by perf_curve_table
// ------------------------------------------------------------------------------------------
using ConfigPerfCurveParams = std::tuple<ov::intel_auto::PerfCurveTable,  // perf_curve_table
                                         std::vector<DeviceInformation>,  // device candidate list
                                         std::map<std::string, float>,   // device utilization (device_name -> value)
                                         DeviceInformation                // expected selected device
                                         >;

class SelectDeviceWithPerfCurveTableTest : public tests::AutoTest,
                                            public ::testing::TestWithParam<ConfigPerfCurveParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigPerfCurveParams> obj) {
        ov::intel_auto::PerfCurveTable perfCurveTable;
        std::vector<ov::auto_plugin::DeviceInformation> devices;
        ov::auto_plugin::DeviceInformation selectedDeviceInfo;
        std::map<std::string, float> deviceUtilization;
        std::tie(perfCurveTable, devices, deviceUtilization, selectedDeviceInfo) = obj.param;
        std::ostringstream result;
        result << "candidateDeviceList_";
        for (const auto& dev : devices)
            result << dev.device_name << "_priority_" << dev.device_priority << "_";
        result << "utilization_";
        for (const auto& item : deviceUtilization) {
            result << item.first << "_" << item.second << "_";
        }
        result << "curveDevices_";
        for (const auto& item : perfCurveTable) {
            result << item.first << "_";
        }
        result << "expectedSelectedDevice_" << selectedDeviceInfo.unique_name;

        auto sanitize_for_gtest = [](std::string name) {
            for (char& ch : name) {
                const unsigned char c = static_cast<unsigned char>(ch);
                if (!std::isalnum(c) && ch != '_') {
                    ch = '_';
                }
            }
            return name;
        };

        return sanitize_for_gtest(result.str());
    }

    void compare(const DeviceInformation& a, const DeviceInformation& b) {
        EXPECT_EQ(a.device_name, b.device_name);
        EXPECT_EQ(a.unique_name, b.unique_name);
        EXPECT_EQ(a.default_device_id, b.default_device_id);
    }

    void SetUp() override {
        std::tie(perfCurveTable, devices, deviceUtilization, selectedDeviceInfo) = GetParam();
        std::vector<std::string> npuCapability = {"FP32", "FP16", "INT8", "BIN"};
        ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_NPU), StrEq(ov::device::capabilities.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(npuCapability));
        ON_CALL(*plugin, get_device_utilization)
            .WillByDefault([this](const std::string& device_name, const std::string& device_type) -> std::optional<float> {
                const auto it = deviceUtilization.find(device_name);
                if (it == deviceUtilization.end()) {
                    return std::nullopt;
                }
                return it->second;
            });
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision);
            });
    }

protected:
    ov::intel_auto::PerfCurveTable perfCurveTable;
    std::vector<ov::auto_plugin::DeviceInformation> devices;
    ov::auto_plugin::DeviceInformation selectedDeviceInfo;
    std::map<std::string, float> deviceUtilization;
};

TEST_P(SelectDeviceWithPerfCurveTableTest, selectDeviceWithPerfCurveTable) {
    std::string netPrecision = "FP32";
    auto result = plugin->select_device(devices, netPrecision, 0, {{}, perfCurveTable}, {});
    compare(result, selectedDeviceInfo);
    // m_priority_map is process-wide static state; clean up to avoid leaking into other suites.
    plugin->unregister_priority(0, result.unique_name);
}

const std::vector<ConfigPerfCurveParams> testPerfCurveConfigs = {
    // 1. Single device, pure interpolation.
    ConfigPerfCurveParams{{{"CPU", {{0, 0.f}, {100, 100.f}}}},
                          {{"CPU", {}, -1, "01", "CPU_01", 0}},
                          {{"CPU", 25.f}},
                          {"CPU", {}, -1, "01", "CPU_01", 0}},
    // 2. Two devices; lower interpolated score wins.
    ConfigPerfCurveParams{{{"CPU", {{0, 0.f}, {100, 100.f}}}, {"NPU", {{0, 0.f}, {100, 100.f}}}},
                          {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}},
                          {{"CPU", 70.f}, {"NPU", 30.f}},
                          {"NPU", {}, -1, "01", "NPU_01", 0}},
    // 3. iGPU/dGPU resolved via ov::device::type.
    ConfigPerfCurveParams{{{"iGPU", {{0, 0.f}, {100, 100.f}}}, {"dGPU", {{0, 100.f}, {100, 0.f}}}},
                          {{"GPU.0", {}, -1, "01", "iGPU_01", 0}, {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                          {{"GPU.0", 80.f}, {"GPU.1", 80.f}},
                          {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
    // 4. No curve entry for a candidate excludes it from ranking.
    ConfigPerfCurveParams{{{"CPU", {{0, 0.f}, {100, 100.f}}}, {"NPU", {{0, 0.f}, {100, 100.f}}}},
                          {{"CPU", {}, -1, "01", "CPU_01", 0},
                           {"NPU", {}, -1, "01", "NPU_01", 0},
                           {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
                          {{"CPU", 60.f}, {"NPU", 20.f}, {"GPU.0", 5.f}},
                          {"NPU", {}, -1, "01", "NPU_01", 0}},
    // 5. No utilization data for a candidate treats it as unscored.
    ConfigPerfCurveParams{{{"CPU", {{0, 0.f}, {100, 100.f}}}, {"NPU", {{0, 0.f}, {100, 100.f}}}},
                          {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}},
                          {{"CPU", 45.f}},
                          {"CPU", {}, -1, "01", "CPU_01", 0}},
    // 6. Out-of-range utilization throws internally; treated as unscored.
    ConfigPerfCurveParams{{{"CPU", {{50, 0.f}, {100, 50.f}}}, {"NPU", {{0, 0.f}, {100, 100.f}}}},
                          {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}},
                          {{"CPU", 10.f}, {"NPU", 70.f}},
                          {"NPU", {}, -1, "01", "NPU_01", 0}},
    // 7. No curve entry matches any device; falls back to priority order.
    ConfigPerfCurveParams{{{"iGPU", {{0, 0.f}, {100, 100.f}}}},
                          {{"CPU", {}, -1, "01", "CPU_01", 1}, {"NPU", {}, -1, "01", "NPU_01", 2}},
                          {{"CPU", 50.f}, {"NPU", 10.f}},
                          {"CPU", {}, -1, "01", "CPU_01", 1}},
    // 8. Fine-grained (10-unit step) curves; interpolation picks the correct bracketing pair.
    ConfigPerfCurveParams{
        {{"CPU",
          {{0, 0.f}, {10, 5.f}, {20, 15.f}, {30, 30.f}, {40, 50.f}, {50, 52.f}, {60, 55.f},
           {70, 60.f}, {80, 70.f}, {90, 85.f}, {100, 100.f}}},
         {"NPU",
          {{0, 100.f}, {10, 90.f}, {20, 80.f}, {30, 70.f}, {40, 60.f}, {50, 50.f}, {60, 40.f},
           {70, 30.f}, {80, 20.f}, {90, 10.f}, {100, 0.f}}}},
        {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}},
        {{"CPU", 45.f}, {"NPU", 75.f}},
        {"NPU", {}, -1, "01", "NPU_01", 0}},
    // 9. CPU + iGPU + NPU, all fine-grained (10-unit step); lowest score among all three wins.
    ConfigPerfCurveParams{
        {{"CPU",
          {{0, 100.f}, {10, 90.f}, {20, 80.f}, {30, 70.f}, {40, 60.f}, {50, 50.f}, {60, 40.f},
           {70, 30.f}, {80, 20.f}, {90, 10.f}, {100, 0.f}}},
         {"iGPU",
          {{0, 80.f}, {10, 60.f}, {20, 40.f}, {30, 20.f}, {40, 10.f}, {50, 5.f}, {60, 10.f},
           {70, 20.f}, {80, 40.f}, {90, 60.f}, {100, 80.f}}},
         {"NPU",
          {{0, 0.f}, {10, 10.f}, {20, 20.f}, {30, 30.f}, {40, 40.f}, {50, 50.f}, {60, 60.f},
           {70, 70.f}, {80, 80.f}, {90, 90.f}, {100, 100.f}}}},
        {{"CPU", {}, -1, "01", "CPU_01", 0},
         {"GPU.0", {}, -1, "01", "iGPU_01", 0},
         {"NPU", {}, -1, "01", "NPU_01", 0}},
        {{"CPU", 65.f}, {"GPU.0", 52.f}, {"NPU", 20.f}},
        {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SelectDeviceWithPerfCurveTableTest,
                         ::testing::ValuesIn(testPerfCurveConfigs),
                         SelectDeviceWithPerfCurveTableTest::getTestCaseName);

// devices_utilization_threshold is applied before perf_curve_table when both are set.
class SelectDeviceThresholdBeforePerfCurveTableTest : public tests::AutoTest, public ::testing::Test {
public:
    void SetUp() override {
        std::vector<std::string> npuCapability = {"FP32", "FP16", "INT8", "BIN"};
        ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_NPU), StrEq(ov::device::capabilities.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(npuCapability));
        ON_CALL(*plugin, get_device_utilization)
            .WillByDefault([this](const std::string& device_name, const std::string& device_type) -> std::optional<float> {
                const auto it = deviceUtilization.find(device_name);
                if (it == deviceUtilization.end()) {
                    return std::nullopt;
                }
                return it->second;
            });
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision);
            });
    }

protected:
    std::map<std::string, float> deviceUtilization = {{"CPU", 90.f}, {"NPU", 10.f}};
};

TEST_F(SelectDeviceThresholdBeforePerfCurveTableTest, thresholdFiltersCandidatesBeforePerfCurveRanking) {
    std::string netPrecision = "FP32";
    std::vector<DeviceInformation> devices = {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}};
    // CPU exceeds this threshold and must be excluded before perf_curve_table ranking.
    std::unordered_map<std::string, unsigned> thresholds = {{"CPU", 50}};
    ov::intel_auto::PerfCurveTable perfCurveTable = {{"CPU", {{0, 0.f}, {100, 0.f}}},
                                                                        {"NPU", {{0, 100.f}, {100, 100.f}}}};

    auto result = plugin->select_device(devices, netPrecision, 0, {thresholds, perfCurveTable}, {});
    EXPECT_EQ(result.unique_name, "NPU_01");
    // m_priority_map is process-wide static state; clean up to avoid leaking into other suites.
    plugin->unregister_priority(0, result.unique_name);
}

TEST_F(SelectDeviceThresholdBeforePerfCurveTableTest, thresholdResultIsKeptWhenPerfCurveDoesNotScore) {
    std::string netPrecision = "FP32";
    std::vector<DeviceInformation> devices = {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}};
    // CPU (utilization 90) exceeds its threshold and is excluded by threshold filtering.
    std::unordered_map<std::string, unsigned> thresholds = {{"CPU", 50}};
    // perf_curve_table covers only iGPU, so remaining candidates are selected by priority.
    ov::intel_auto::PerfCurveTable perfCurveTable = {{"iGPU", {{0, 0.f}, {100, 100.f}}}};

    auto result = plugin->select_device(devices, netPrecision, 0, {thresholds, perfCurveTable}, {});
    EXPECT_EQ(result.unique_name, "NPU_01");
    plugin->unregister_priority(0, result.unique_name);
}

// ------------------------------------------------------------------------------------------
// sort_device_by_perf_curve() unit tests: verify score computation (exact match, interpolation),
// stable ordering (scored-ascending first, unscored trailing in original relative order), and
// graceful handling of out-of-range utilization.
// ------------------------------------------------------------------------------------------
using ConfigSortByPerfCurveParams = std::tuple<ov::intel_auto::PerfCurveTable,  // perf_curve_table
                                               std::list<DeviceInformation>,  // input device list (order matters)
                                               std::map<std::string, float>,  // device utilization
                                               std::vector<std::string>       // expected unique_name order
                                               >;

class SortDeviceByPerfCurveTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigSortByPerfCurveParams> {
public:
    void SetUp() override {
        std::tie(perfCurveTable, inputDevices, deviceUtilization, expectedOrder) = GetParam();
        ON_CALL(*plugin, get_device_utilization)
            .WillByDefault([this](const std::string& device_name, const std::string& device_type) -> std::optional<float> {
                const auto it = deviceUtilization.find(device_name);
                if (it == deviceUtilization.end()) {
                    return std::nullopt;
                }
                return it->second;
            });
    }

protected:
    ov::intel_auto::PerfCurveTable perfCurveTable;
    std::list<DeviceInformation> inputDevices;
    std::map<std::string, float> deviceUtilization;
    std::vector<std::string> expectedOrder;
};

TEST_P(SortDeviceByPerfCurveTest, sortDeviceByPerfCurve) {
    auto result = plugin->sort_device_by_perf_curve(inputDevices, perfCurveTable, nullptr);
    std::vector<std::string> actualOrder;
    for (const auto& device : result) {
        actualOrder.push_back(device.unique_name);
    }
    EXPECT_EQ(actualOrder, expectedOrder);
}

const std::vector<ConfigSortByPerfCurveParams> testSortByPerfCurveConfigs = {
    // 1. Exact key match: utilization equals a curve key, no interpolation needed.
    ConfigSortByPerfCurveParams{{{"CPU", {{0, 0.f}, {50, 50.f}, {100, 100.f}}}},
                                {{"CPU", {}, -1, "01", "CPU_01", 0}},
                                {{"CPU", 50.f}},
                                {"CPU_01"}},
    // 2. Linear interpolation ratio determines relative ordering.
    ConfigSortByPerfCurveParams{{{"CPU", {{0, 0.f}, {100, 40.f}}}, {"NPU", {{0, 0.f}, {100, 100.f}}}},
                                {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}},
                                {{"CPU", 25.f}, {"NPU", 5.f}},
                                {"NPU_01", "CPU_01"}},
    // 3. Three scored devices sorted ascending by interpolated score.
    ConfigSortByPerfCurveParams{{{"CPU", {{0, 0.f}, {100, 100.f}}},
                                 {"NPU", {{0, 0.f}, {100, 100.f}}},
                                 {"iGPU", {{0, 0.f}, {100, 100.f}}}},
                                {{"CPU", {}, -1, "01", "CPU_01", 0},
                                 {"NPU", {}, -1, "01", "NPU_01", 0},
                                 {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
                                {{"CPU", 30.f}, {"NPU", 10.f}, {"GPU.0", 20.f}},
                                {"NPU_01", "iGPU_01", "CPU_01"}},
    // 4. Unscored devices trail scored ones, keeping their original relative order.
    ConfigSortByPerfCurveParams{{{"CPU", {{0, 0.f}, {100, 100.f}}}},
                                {{"CPU", {}, -1, "01", "CPU_01", 0},
                                 {"NPU", {}, -1, "01", "NPU_01", 0},
                                 {"GPU.0", {}, -1, "01", "iGPU_01", 0}},
                                {{"CPU", 40.f}},
                                {"CPU_01", "NPU_01", "iGPU_01"}},
    // 5. Out-of-range utilization throws internally; treated as unscored (trailing).
    ConfigSortByPerfCurveParams{{{"CPU", {{50, 0.f}, {100, 50.f}}}, {"NPU", {{0, 0.f}, {100, 100.f}}}},
                                {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}},
                                {{"CPU", 10.f}, {"NPU", 60.f}},
                                {"NPU_01", "CPU_01"}},
    // 6. iGPU/dGPU curve resolution affects final ordering.
    ConfigSortByPerfCurveParams{{{"iGPU", {{0, 0.f}, {100, 100.f}}}, {"dGPU", {{0, 100.f}, {100, 0.f}}}},
                                {{"GPU.0", {}, -1, "01", "iGPU_01", 0}, {"GPU.1", {}, -1, "01", "dGPU_01", 0}},
                                {{"GPU.0", 80.f}, {"GPU.1", 80.f}},
                                {"dGPU_01", "iGPU_01"}},
    // 7. Boundary: utilization equals max_key for both devices -> score is the last curve value
    //    (upper_bound == end() path). Ordering reflects those last values (30 < 90).
    ConfigSortByPerfCurveParams{{{"CPU", {{0, 0.f}, {100, 90.f}}}, {"NPU", {{0, 0.f}, {100, 30.f}}}},
                                {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}},
                                {{"CPU", 100.f}, {"NPU", 100.f}},
                                {"NPU_01", "CPU_01"}},
    // 8. Mid-range interpolation between two non-zero keys combined with a max_key hit.
    //    CPU: ratio (50-20)/(80-20)=0.5 -> 10 + 0.5*(70-10) = 40; NPU at max_key -> 50. 40 < 50.
    ConfigSortByPerfCurveParams{{{"CPU", {{20, 10.f}, {80, 70.f}}}, {"NPU", {{0, 0.f}, {100, 50.f}}}},
                                {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}},
                                {{"CPU", 50.f}, {"NPU", 100.f}},
                                {"CPU_01", "NPU_01"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SortDeviceByPerfCurveTest,
                         ::testing::ValuesIn(testSortByPerfCurveConfigs),
                         ::testing::PrintToStringParamName());

// -----------------------------------------------------------------------------------
// ov::intel_auto::low_power_device must take precedence over
// devices_utilization_threshold whenever the platform is reported as being in low power mode.
// get_low_power_mode() is mocked directly to keep this suite deterministic and independent
// from runtime IPF/DTT event delivery.
// -----------------------------------------------------------------------------------
class SelectDeviceWithLowPowerDevicePrecedenceTest : public tests::AutoTest, public ::testing::Test {
public:
    void SetUp() override {
        std::vector<std::string> npuCapability = {"FP32", "FP16", "INT8", "BIN"};
        ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_NPU), StrEq(ov::device::capabilities.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(npuCapability));
        ON_CALL(*plugin, get_device_utilization)
            .WillByDefault([this](const std::string& device_name, const std::string& device_type) -> std::optional<float> {
                const auto it = deviceUtilization.find(device_name);
                if (it == deviceUtilization.end()) {
                    return std::nullopt;
                }
                return it->second;
            });
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                return plugin->Plugin::get_valid_device(metaDevices, netPrecision);
            });
    }

    void TearDown() override {
        // m_priority_map is process-wide static state; clean up to avoid leaking into other suites.
        if (!selectedUniqueName.empty()) {
            plugin->unregister_priority(0, selectedUniqueName);
        }
    }

protected:
    std::string netPrecision = "FP32";
    std::vector<DeviceInformation> devices = {{"CPU", {}, -1, "01", "CPU_01", 0}, {"NPU", {}, -1, "01", "NPU_01", 0}};
    // NPU exceeds this threshold and is excluded by threshold logic; only low_power_device can pick it.
    std::unordered_map<std::string, unsigned> thresholds = {{"NPU", 50}};
    std::map<std::string, float> deviceUtilization = {{"CPU", 10.f}, {"NPU", 90.f}};
    std::string selectedUniqueName;
};

TEST_F(SelectDeviceWithLowPowerDevicePrecedenceTest, lowPowerDeviceOverridesThreshold) {
    EXPECT_CALL(*plugin, get_low_power_mode()).WillOnce(Return(true));
    auto result = plugin->select_device(devices, netPrecision, 0, {thresholds, {}}, "NPU");
    selectedUniqueName = result.unique_name;
    EXPECT_EQ(result.unique_name, "NPU_01");
}

TEST_F(SelectDeviceWithLowPowerDevicePrecedenceTest, lowPowerDeviceMatchesViaBaseNameFallback) {
    // low_power_device="NPU" must still match a candidate whose device_name carries a HW id suffix.
    std::vector<std::string> npuHwIdCapability = {"FP32", "FP16", "INT8", "BIN"};
    ON_CALL(*core, get_property(StrEq("NPU.5010"), StrEq(ov::device::capabilities.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(npuHwIdCapability));
    std::vector<DeviceInformation> devicesWithHwId = {{"CPU", {}, -1, "01", "CPU_01", 0},
                                                       {"NPU.5010", {}, -1, "01", "NPU_01", 0}};
    EXPECT_CALL(*plugin, get_low_power_mode()).WillOnce(Return(true));
    auto result = plugin->select_device(devicesWithHwId, netPrecision, 0, {thresholds, {}}, "NPU");
    selectedUniqueName = result.unique_name;
    EXPECT_EQ(result.unique_name, "NPU_01");
}

TEST_F(SelectDeviceWithLowPowerDevicePrecedenceTest, fallsBackToThresholdWhenNotInLowPowerMode) {
    EXPECT_CALL(*plugin, get_low_power_mode()).WillOnce(Return(false));
    auto result = plugin->select_device(devices, netPrecision, 0, {thresholds, {}}, "NPU");
    selectedUniqueName = result.unique_name;
    // threshold logic excludes the over-utilized NPU once low_power_device is not applicable.
    EXPECT_EQ(result.unique_name, "CPU_01");
}

TEST_F(SelectDeviceWithLowPowerDevicePrecedenceTest, unknownLowPowerModeTreatedAsNotLowPower) {
    EXPECT_CALL(*plugin, get_low_power_mode()).WillOnce(Return(std::nullopt));
    auto result = plugin->select_device(devices, netPrecision, 0, {thresholds, {}}, "NPU");
    selectedUniqueName = result.unique_name;
    EXPECT_EQ(result.unique_name, "CPU_01");
}

TEST_F(SelectDeviceWithLowPowerDevicePrecedenceTest, lowPowerDeviceNotInCandidateListFallsThrough) {
    // get_low_power_mode() is not called when the preferred device is not a candidate.
    EXPECT_CALL(*plugin, get_low_power_mode()).Times(0);
    auto result = plugin->select_device(devices, netPrecision, 0, {thresholds, {}}, "GPU.0");
    selectedUniqueName = result.unique_name;
    EXPECT_EQ(result.unique_name, "CPU_01");
}

TEST_F(SelectDeviceWithLowPowerDevicePrecedenceTest, getLowPowerModeNotQueriedWhenLowPowerDeviceUnset) {
    // Guarantees zero behavior/perf impact on all pre-existing callers that leave low_power_device empty.
    EXPECT_CALL(*plugin, get_low_power_mode()).Times(0);
    auto result = plugin->select_device(devices, netPrecision, 0, {thresholds, {}}, {});
    selectedUniqueName = result.unique_name;
    EXPECT_EQ(result.unique_name, "CPU_01");
}

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
                                  unsigned int priority) {
                return plugin->Plugin::select_device(metaDevices, netPrecision, priority);
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

    EXPECT_CALL(*plugin, select_device(_, _, _)).Times(1);
    if (devices.size() >= 1) {
        EXPECT_CALL(*core, get_property(_, _, _)).Times(AtLeast(static_cast<int>(devices.size()) - 1));
    } else {
        EXPECT_CALL(*core, get_property(_, _, _)).Times(0);
    }

    if (throwExcept) {
        ASSERT_THROW(plugin->select_device(devices, netPrecision, 0), ov::Exception);
    } else {
        auto result = plugin->select_device(devices, netPrecision, 0);
        compare(result, expect);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SelectDeviceTest,
                         ::testing::ValuesIn(SelectDeviceTest::CreateConfigs()),
                         SelectDeviceTest::getTestCaseName);
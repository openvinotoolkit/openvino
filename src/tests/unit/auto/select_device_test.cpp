// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <common_test_utils/test_constants.hpp>
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "plugin/mock_auto_device_plugin.hpp"
#include "cpp/ie_plugin.hpp"
#include "mock_common.hpp"

using ::testing::MatcherCast;
using ::testing::AllOf;
using ::testing::Throw;
using ::testing::Matches;
using ::testing::_;
using ::testing::StrEq;
using ::testing::Return;
using ::testing::Property;
using ::testing::Eq;
using ::testing::ReturnRef;
using ::testing::AtLeast;
using ::testing::InvokeWithoutArgs;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

using ConfigParams = std::tuple<
        std::string,                        // netPrecision
        std::vector<DeviceInformation>,      // metaDevices for select
        DeviceInformation,                   // expect DeviceInformation
        bool,                                // throw exception
        bool,                                // enableDevicePriority
        bool                                 // reverse total device
        >;

const DeviceInformation CPU_INFO = {CommonTestUtils::DEVICE_CPU, {}, 2, "01", "CPU_01"};
const DeviceInformation IGPU_INFO = {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "iGPU_01"};
const DeviceInformation DGPU_INFO = {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "dGPU_01"};
const DeviceInformation MYRIAD_INFO = {CommonTestUtils::DEVICE_MYRIAD, {}, 2, "01", "MYRIAD_01" };
const DeviceInformation KEEMBAY_INFO = {CommonTestUtils::DEVICE_KEEMBAY, {}, 2, "01", "VPUX_01" };
const std::vector<DeviceInformation>  fp32DeviceVector = {DGPU_INFO, IGPU_INFO, CPU_INFO, MYRIAD_INFO};
const std::vector<DeviceInformation>  fp16DeviceVector = {DGPU_INFO, IGPU_INFO, MYRIAD_INFO, CPU_INFO};
const std::vector<DeviceInformation>  int8DeviceVector = {KEEMBAY_INFO, DGPU_INFO, IGPU_INFO, CPU_INFO};
const std::vector<DeviceInformation>  binDeviceVector = {DGPU_INFO, IGPU_INFO, CPU_INFO};
const std::vector<DeviceInformation>  batchedblobDeviceVector = {DGPU_INFO, IGPU_INFO};
std::map<std::string, const std::vector<DeviceInformation>> devicesMap = {{"FP32", fp32DeviceVector},
                                                                           {"FP16", fp16DeviceVector},
                                                                           {"INT8", int8DeviceVector},
                                                                           {"BIN",  binDeviceVector},
                                                                           {"BATCHED_BLOB", batchedblobDeviceVector}
                                                                         };
const std::vector<DeviceInformation> totalDevices = {DGPU_INFO, IGPU_INFO, MYRIAD_INFO, CPU_INFO, KEEMBAY_INFO};
const std::vector<DeviceInformation> reverseTotalDevices = {KEEMBAY_INFO, CPU_INFO, MYRIAD_INFO, IGPU_INFO, DGPU_INFO};
const std::vector<std::string> netPrecisions = {"FP32", "FP16", "INT8", "BIN", "BATCHED_BLOB"};
std::vector<ConfigParams> testConfigs;

class SelectDeviceTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<MockICore>                      core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string netPrecision;
        std::vector<DeviceInformation> devices;
        DeviceInformation expect;
        bool throwExcept;
        bool enableDevicePriority;
        bool reverse;
        std::tie(netPrecision, devices, expect, throwExcept, enableDevicePriority, reverse) = obj.param;
        std::ostringstream result;
        result << "_netPrecision_" << netPrecision;
        for (auto& item : devices) {
            result <<  "_device_" << item.uniqueName;
        }
        result << "_expect_" << expect.uniqueName;
        if (throwExcept) {
            result << "_throwExcept_true";
        } else {
            result << "_throwExcept_false";
        }

        if (enableDevicePriority) {
            result << "_enableDevicePriority_true";
        } else {
            result << "_enableDevicePriority_false";
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
    static void combine_device(const std::vector<DeviceInformation>& devices, int start,
            int* result, int result_index, const int select_num, std::string& netPrecision,
            bool enableDevicePriority, bool reverse) {
        int i = 0;
        for (i = start; i < devices.size() + 1 - result_index; i++) {
            result[result_index - 1] = i;
            if (result_index - 1 == 0) {
                std::vector<DeviceInformation> metaDevices = {};
                int devicePriority = 0;
                for (int j = select_num - 1; j >= 0; j--) {
                    auto tmpDevInfo = devices[result[j]];
                    if (enableDevicePriority) {
                        tmpDevInfo.devicePriority = devicePriority;
                        devicePriority++;
                    }
                    metaDevices.push_back(tmpDevInfo);
                }
                // Debug the combine_device
                // for (auto& item : metaDevices) {
                //     std::cout << item.uniqueName << "_";
                // }
                // std::cout << netPrecision << std::endl;
                auto& devicesInfo = devicesMap[netPrecision];
                bool find = false;
                DeviceInformation expect;
                if (metaDevices.size() > 1) {
                    if (enableDevicePriority) {
                        std::vector<DeviceInformation> validDevices;
                        for (auto& item : devicesInfo) {
                            auto device =  std::find_if(metaDevices.begin(), metaDevices.end(),
                                    [&item](const DeviceInformation& d)->bool{return d.uniqueName == item.uniqueName;});
                            if (device != metaDevices.end()) {
                                validDevices.push_back(*device);
                            }
                        }
                        int currentDevicePriority = 100;
                        for (auto iter = validDevices.begin(); iter != validDevices.end(); iter++) {
                            if (iter->devicePriority < currentDevicePriority) {
                                expect = *iter;
                                currentDevicePriority = iter->devicePriority;
                            }
                        }
                        if (currentDevicePriority != 100) {
                            find = true;
                        }
                    } else {
                        for (auto& item : devicesInfo) {
                            auto device =  std::find_if(metaDevices.begin(), metaDevices.end(),
                                    [&item](const DeviceInformation& d)->bool{return d.uniqueName == item.uniqueName;});
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
                testConfigs.push_back(std::make_tuple(netPrecision, metaDevices,
                            expect, !find, enableDevicePriority, reverse));
            } else {
                combine_device(devices, i + 1, result, result_index - 1,
                        select_num, netPrecision, enableDevicePriority, reverse);
            }
        }
    }

    static std::vector<ConfigParams> CreateConfigs() {
        auto result = new int[totalDevices.size()];
        // test all netPrecision with all possible combine devices
        // netPrecision number is 5
        // device number is 5
        // combine devices is 5!/5! + 5!/(4!*1!) + 5!/(3!*2!) + 5!/(2!*3!) + 5(1!*4!) = 31
        // null device 1
        // total test config num is 32*5 = 160
        for (auto netPrecision : netPrecisions) {
            for (int i = 1; i <= totalDevices.size(); i++) {
                combine_device(totalDevices, 0, result, i, i, netPrecision, false, false);
            }
            // test null device
            testConfigs.push_back(ConfigParams{netPrecision, {}, {}, true, false, false});
        }
        // reverse totalDevices for test
        for (auto netPrecision : netPrecisions) {
            for (int i = 1; i <= reverseTotalDevices.size(); i++) {
                combine_device(reverseTotalDevices, 0, result, i, i, netPrecision, false, true);
            }
        }

        // add test for enableDevicePriority
        // test case num is 31*5 = 155
        for (auto netPrecision : netPrecisions) {
            for (int i = 1; i <= totalDevices.size(); i++) {
                combine_device(totalDevices, 0, result, i, i, netPrecision, true, false);
            }
        }

        // reverse totalDevices for test
        for (auto netPrecision : netPrecisions) {
            for (int i = 1; i <= reverseTotalDevices.size(); i++) {
                combine_device(reverseTotalDevices, 0, result, i, i, netPrecision, true, true);
            }
        }
        delete []result;
        return testConfigs;
    }

    void compare(DeviceInformation& a, DeviceInformation& b) {
        EXPECT_EQ(a.deviceName, b.deviceName);
        EXPECT_EQ(a.uniqueName, b.uniqueName);
        EXPECT_EQ(a.defaultDeviceID, b.defaultDeviceID);
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
       // prepare mockicore and cnnNetwork for loading
       core  = std::shared_ptr<MockICore>(new MockICore());
       auto* origin_plugin = new MockMultiDeviceInferencePlugin();
       plugin  = std::shared_ptr<MockMultiDeviceInferencePlugin>(origin_plugin);
       // replace core with mock Icore
       plugin->SetCore(core);

       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, cpuCability, {"FP32", "FP16", "INT8", "BIN"});
       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, gpuCability, {"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"});
       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, myriadCability, {"FP16"});
       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, vpuxCability, {"INT8"});

       ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_CPU),
                   StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _)).WillByDefault(RETURN_MOCK_VALUE(cpuCability));
       ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_GPU),
                   StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _)).WillByDefault(RETURN_MOCK_VALUE(gpuCability));
       ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_MYRIAD),
                   StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _)).WillByDefault(RETURN_MOCK_VALUE(myriadCability));
       ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_KEEMBAY),
                   StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _)).WillByDefault(RETURN_MOCK_VALUE(vpuxCability));
       ON_CALL(*plugin, SelectDevice).WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                   const std::string& netPrecision, unsigned int priority) {
               return plugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, priority);
               });
    }
};

TEST_P(SelectDeviceTest, SelectDevice) {
    // get Parameter
    std::string netPrecision;
    std::vector<DeviceInformation> devices;
    DeviceInformation expect;
    bool throwExcept;
    bool enableDevicePriority;
    bool reverse;
    std::tie(netPrecision, devices, expect, throwExcept, enableDevicePriority, reverse) = this->GetParam();

    EXPECT_CALL(*plugin, SelectDevice(_, _, _)).Times(1);
    if (devices.size() >= 1) {
        EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AtLeast(devices.size() - 1));
    } else {
        EXPECT_CALL(*core, GetMetric(_, _, _)).Times(0);
    }

    if (throwExcept) {
        ASSERT_THROW(plugin->SelectDevice(devices, netPrecision, 0), InferenceEngine::Exception);
    } else {
        auto result =  plugin->SelectDevice(devices, netPrecision, 0);
        compare(result, expect);
    }
}



INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, SelectDeviceTest,
                ::testing::ValuesIn(SelectDeviceTest::CreateConfigs()),
            SelectDeviceTest::getTestCaseName);


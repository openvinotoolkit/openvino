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

using PriorityParams = std::tuple<unsigned int, std::string>; //{modelpriority, deviceUniquName}

using ConfigParams = std::tuple<
        std::string,                       // netPrecision
        bool,                              // enable device priority
        std::vector<PriorityParams>        // {{modelpriority, expect device uniqueName}}
        >;
class KeyNetworkPriorityTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<MockICore>                      core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;
    std::vector<DeviceInformation>                  metaDevices;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string netPrecision;
        bool enableDevicePriority;
        std::vector<PriorityParams> PriorityConfigs;
        std::tie(netPrecision, enableDevicePriority, PriorityConfigs) = obj.param;
        std::ostringstream result;
        if (enableDevicePriority) {
            result << "_enableDevicePriority_true";
        } else {
            result << "_enableDevicePriority_false";
        }
        for (auto& item : PriorityConfigs) {
            result <<  "_priority_" << std::get<0>(item);
            result <<  "_return_" << std::get<1>(item);
        }
        result << "netPrecision_" << netPrecision;
        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        metaDevices.clear();
    }

    void SetUp() override {
       // prepare mockicore and cnnNetwork for loading
       core  = std::shared_ptr<MockICore>(new MockICore());
       auto* origin_plugin = new MockMultiDeviceInferencePlugin();
       plugin  = std::shared_ptr<MockMultiDeviceInferencePlugin>(origin_plugin);
       // replace core with mock Icore
       plugin->SetCore(core);

       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, cpuCability, {"FP32", "FP16", "INT8", "BIN"});
       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, gpuCability, {"FP32", "FP16", "BATCHED_BLOB", "BIN"});
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
                   const std::string& netPrecision, unsigned int Priority) {
               return plugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, Priority);
               });
    }
};

TEST_P(KeyNetworkPriorityTest, SelectDevice) {
    // get Parameter

    std::string netPrecision;
    bool enableDevicePriority;
    std::vector<PriorityParams> PriorityConfigs;
    std::tie(netPrecision, enableDevicePriority, PriorityConfigs) = this->GetParam();
    std::vector<DeviceInformation> resDevInfo;

    if (enableDevicePriority) {
        metaDevices = {{CommonTestUtils::DEVICE_CPU, {}, 2, "", "CPU_01", 0},
            {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "iGPU_01", 1},
            {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "dGPU_01", 2},
            {CommonTestUtils::DEVICE_MYRIAD, {}, 2, "01", "MYRIAD_01", 3},
            {CommonTestUtils::DEVICE_KEEMBAY, {}, 2, "01", "VPUX_01", 4}};
    } else {
        metaDevices = {{CommonTestUtils::DEVICE_CPU, {}, 2, "", "CPU_01", 0},
            {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "iGPU_01", 0},
            {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "dGPU_01", 0},
            {CommonTestUtils::DEVICE_MYRIAD, {}, 2, "01", "MYRIAD_01", 0},
            {CommonTestUtils::DEVICE_KEEMBAY, {}, 2, "01", "VPUX_01", 0}};
    }

    EXPECT_CALL(*plugin, SelectDevice(_, _, _)).Times(PriorityConfigs.size());
    EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AtLeast(PriorityConfigs.size() * 4));

    for (auto& item : PriorityConfigs) {
        resDevInfo.push_back(plugin->SelectDevice(metaDevices, netPrecision, std::get<0>(item)));
    }
    for (unsigned int i = 0; i < PriorityConfigs.size(); i++) {
        EXPECT_EQ(resDevInfo[i].uniqueName, std::get<1>(PriorityConfigs[i]));
        plugin->UnregisterPriority(std::get<0>(PriorityConfigs[i]), std::get<1>(PriorityConfigs[i]));
    }
}

TEST_P(KeyNetworkPriorityTest, MultiThreadsSelectDevice) {
    // get Parameter
    std::string netPrecision;
    bool enableDevicePriority;
    std::vector<PriorityParams> PriorityConfigs;
    std::tie(netPrecision, enableDevicePriority, PriorityConfigs) = this->GetParam();
    std::vector<DeviceInformation> resDevInfo;
    std::vector<std::future<void>> futureVect;

    if (enableDevicePriority) {
        metaDevices = {{CommonTestUtils::DEVICE_CPU, {}, 2, "", "CPU_01", 0},
            {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "iGPU_01", 1},
            {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "dGPU_01", 2},
            {CommonTestUtils::DEVICE_MYRIAD, {}, 2, "01", "MYRIAD_01", 3},
            {CommonTestUtils::DEVICE_KEEMBAY, {}, 2, "01", "VPUX_01", 4}};
    } else {
        metaDevices = {{CommonTestUtils::DEVICE_CPU, {}, 2, "", "CPU_01", 0},
            {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "iGPU_01", 0},
            {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "dGPU_01", 0},
            {CommonTestUtils::DEVICE_MYRIAD, {}, 2, "01", "MYRIAD_01", 0},
            {CommonTestUtils::DEVICE_KEEMBAY, {}, 2, "01", "VPUX_01", 0}};
    }

    EXPECT_CALL(*plugin, SelectDevice(_, _, _)).Times(PriorityConfigs.size() * 2);
    EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AtLeast(PriorityConfigs.size() * 4 * 2));
    // selectdevice in multi threads, and UnregisterPriority them all, should not affect the
    // Priority Map
    for (auto& item : PriorityConfigs) {
       unsigned int priority = std::get<0>(item);
       auto future = std::async(std::launch::async, [this, &netPrecision, priority] {
               auto deviceInfo = plugin->SelectDevice(metaDevices, netPrecision, priority);
               plugin->UnregisterPriority(priority, deviceInfo.uniqueName);
               });
       futureVect.push_back(std::move(future));
    }

    for (auto& item : futureVect) {
           item.get();
    }

    for (auto& item : PriorityConfigs) {
        resDevInfo.push_back(plugin->SelectDevice(metaDevices, netPrecision, std::get<0>(item)));
    }
    for (unsigned int i = 0; i < PriorityConfigs.size(); i++) {
        EXPECT_EQ(resDevInfo[i].uniqueName, std::get<1>(PriorityConfigs[i]));
        plugin->UnregisterPriority(std::get<0>(PriorityConfigs[i]), std::get<1>(PriorityConfigs[i]));
    }
}


// ConfigParams details
// example
// ConfigParams {"FP32", false, {PriorityParams {0, "dGPU_01"},
//                        PriorityParams {1, "iGPU_01"},
//                        PriorityParams {2, "MYRIAD_01"},
//                        PriorityParams {2, "MYRIAD_01"}}},
//              {netPrecision, enableDevicePriority,  PriorityParamsVector{{modelpriority, expect device uniqueName}}}

const std::vector<ConfigParams> testConfigs = {
    ConfigParams {"FP32", false, {PriorityParams {0, "dGPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {2, "CPU_01"}}},
    ConfigParams {"FP32", false, {PriorityParams {2, "dGPU_01"},
        PriorityParams {3, "iGPU_01"},
        PriorityParams {4, "CPU_01"},
        PriorityParams {5, "MYRIAD_01"}}},
    ConfigParams {"FP32", false, {PriorityParams {2, "dGPU_01"},
        PriorityParams {0, "dGPU_01"},
        PriorityParams {2, "iGPU_01"},
        PriorityParams {2, "iGPU_01"}}},
    ConfigParams {"FP32", false, {PriorityParams {2, "dGPU_01"},
        PriorityParams {0, "dGPU_01"},
        PriorityParams {2, "iGPU_01"},
        PriorityParams {3, "CPU_01"}}},
    ConfigParams {"FP32", false, {PriorityParams {0, "dGPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {3, "MYRIAD_01"},
        PriorityParams {0, "dGPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {3, "MYRIAD_01"}}},
    ConfigParams {"INT8", false, {PriorityParams {0, "VPUX_01"},
        PriorityParams {1, "CPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {2, "CPU_01"}}},
    ConfigParams {"INT8", false, {PriorityParams {2, "VPUX_01"},
        PriorityParams {3, "CPU_01"},
        PriorityParams {4, "CPU_01"},
        PriorityParams {5, "CPU_01"}}},
    ConfigParams {"INT8", false, {PriorityParams {2, "VPUX_01"},
        PriorityParams {0, "VPUX_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {2, "CPU_01"}}},
    ConfigParams {"INT8", false, {PriorityParams {2, "VPUX_01"},
        PriorityParams {0, "VPUX_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {3, "CPU_01"}}},
    ConfigParams {"INT8", false, {PriorityParams {0, "VPUX_01"},
        PriorityParams {1, "CPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {3, "CPU_01"},
        PriorityParams {0, "VPUX_01"},
        PriorityParams {1, "CPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {3, "CPU_01"}}},
    ConfigParams {"BIN", false, {PriorityParams {0, "dGPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {2, "CPU_01"}}},
    ConfigParams {"BIN", false, {PriorityParams {2, "dGPU_01"},
        PriorityParams {3, "iGPU_01"},
        PriorityParams {4, "CPU_01"},
        PriorityParams {5, "CPU_01"}}},
    ConfigParams {"BIN", false, {PriorityParams {2, "dGPU_01"},
        PriorityParams {0, "dGPU_01"},
        PriorityParams {2, "iGPU_01"},
        PriorityParams {2, "iGPU_01"}}},
    ConfigParams {"BIN", false, {PriorityParams {2, "dGPU_01"},
        PriorityParams {0, "dGPU_01"},
        PriorityParams {2, "iGPU_01"},
        PriorityParams {3, "CPU_01"}}},
    ConfigParams {"BIN", false, {PriorityParams {0, "dGPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {3, "CPU_01"},
        PriorityParams {0, "dGPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "CPU_01"},
        PriorityParams {3, "CPU_01"}}},
    // metaDevices = {{CommonTestUtils::DEVICE_CPU, {}, 2, "", "CPU_01", 0},
    // {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "iGPU_01", 1},
    // {CommonTestUtils::DEVICE_GPU, {}, 2, "01", "dGPU_01", 2},
    // {CommonTestUtils::DEVICE_MYRIAD, {}, 2, "01", "MYRIAD_01", 3},
    // {CommonTestUtils::DEVICE_KEEMBAY, {}, 2, "01", "VPUX_01", 4}};
    // cpu > igpu > dgpu > MYRIAD > VPUX
    ConfigParams {"FP32", true, {PriorityParams {0, "CPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "dGPU_01"},
        PriorityParams {2, "dGPU_01"}}},
    ConfigParams {"FP32", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {3, "iGPU_01"},
        PriorityParams {4, "dGPU_01"},
        PriorityParams {5, "MYRIAD_01"}}},
    ConfigParams {"FP32", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {2, "iGPU_01"},
        PriorityParams {2, "iGPU_01"}}},
    ConfigParams {"FP32", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {2, "iGPU_01"},
        PriorityParams {3, "dGPU_01"}}},
    ConfigParams {"FP32", true, {PriorityParams {0, "CPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "dGPU_01"},
        PriorityParams {3, "MYRIAD_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "dGPU_01"},
        PriorityParams {3, "MYRIAD_01"}}},
    ConfigParams {"INT8", true, {PriorityParams {0, "CPU_01"},
        PriorityParams {1, "VPUX_01"},
        PriorityParams {2, "VPUX_01"},
        PriorityParams {2, "VPUX_01"}}},
    ConfigParams {"INT8", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {3, "VPUX_01"},
        PriorityParams {4, "VPUX_01"},
        PriorityParams {5, "VPUX_01"}}},
    ConfigParams {"INT8", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {2, "VPUX_01"},
        PriorityParams {2, "VPUX_01"}}},
    ConfigParams {"INT8", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {2, "VPUX_01"},
        PriorityParams {3, "VPUX_01"}}},
    ConfigParams {"INT8", true, {PriorityParams {0, "CPU_01"},
        PriorityParams {1, "VPUX_01"},
        PriorityParams {2, "VPUX_01"},
        PriorityParams {3, "VPUX_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {1, "VPUX_01"},
        PriorityParams {2, "VPUX_01"},
        PriorityParams {3, "VPUX_01"}}},
    ConfigParams {"BIN", true, {PriorityParams {0, "CPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "dGPU_01"},
        PriorityParams {2, "dGPU_01"}}},
    ConfigParams {"BIN", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {3, "iGPU_01"},
        PriorityParams {4, "dGPU_01"},
        PriorityParams {5, "dGPU_01"}}},
    ConfigParams {"BIN", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {2, "iGPU_01"},
        PriorityParams {2, "iGPU_01"}}},
    ConfigParams {"BIN", true, {PriorityParams {2, "CPU_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {2, "iGPU_01"},
        PriorityParams {3, "dGPU_01"}}},
    ConfigParams {"BIN", true, {PriorityParams {0, "CPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "dGPU_01"},
        PriorityParams {3, "dGPU_01"},
        PriorityParams {0, "CPU_01"},
        PriorityParams {1, "iGPU_01"},
        PriorityParams {2, "dGPU_01"},
        PriorityParams {3, "dGPU_01"}}}
};


INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, KeyNetworkPriorityTest,
                ::testing::ValuesIn(testConfigs),
            KeyNetworkPriorityTest::getTestCaseName);


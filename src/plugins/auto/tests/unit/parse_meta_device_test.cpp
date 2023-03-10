// Copyright (C) 2018-2023 Intel Corporation
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
#include "include/mock_auto_device_plugin.hpp"
#include "include/mock_common.hpp"

using ::testing::MatcherCast;
using ::testing::HasSubstr;
using ::testing::AllOf;
using ::testing::Throw;
using ::testing::Matches;
using ::testing::_;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Return;
using ::testing::Property;
using ::testing::Eq;
using ::testing::AnyNumber;
using ::testing::ReturnRef;
using ::testing::AtLeast;
using ::testing::InvokeWithoutArgs;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

// const char cpuFullDeviceName[] = "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz";
const char igpuFullDeviceName[] = "Intel(R) Gen9 HD Graphics (iGPU)";
const char dgpuFullDeviceName[] = "Intel(R) Iris(R) Xe MAX Graphics (dGPU)";
// const char myriadFullDeviceName[] = "Intel Movidius Myriad X VPU";
// const char vpuxFullDeviceName[] = "";
const std::vector<std::string>  availableDevs = {"CPU", "GPU.0", "GPU.1", "VPUX"};
const std::vector<std::string>  availableDevsNoID = {"CPU", "GPU", "VPUX"};
using ConfigParams = std::tuple<
        std::string,                        // Priority devices
        std::vector<DeviceInformation>,     // expect metaDevices
        bool                                // if throw exception
        >;
class ParseMetaDeviceTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<MockICore>                      core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string priorityDevices;
        std::vector<DeviceInformation> metaDevices;
        bool throwException;
        std::tie(priorityDevices, metaDevices, throwException) = obj.param;
        std::ostringstream result;
        result << "priorityDevices_" << priorityDevices;
        if (throwException) {
            result << "_throwException_true";
        } else {
            result << "_throwException_false";
        }
        return result.str();
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

       IE_SET_METRIC(SUPPORTED_METRICS, metrics, {METRIC_KEY(SUPPORTED_CONFIG_KEYS), METRIC_KEY(FULL_DEVICE_NAME)});
       ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _))
           .WillByDefault(RETURN_MOCK_VALUE(metrics));
       ON_CALL(*core, GetMetric(StrEq("GPU"),
                   StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(igpuFullDeviceName));
       ON_CALL(*core, GetConfig(StrEq("GPU"),
                   StrEq(CONFIG_KEY(DEVICE_ID)))).WillByDefault(Return(0));
       ON_CALL(*core, GetMetric(StrEq("GPU.0"),
                   StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(igpuFullDeviceName));
       ON_CALL(*core, GetMetric(StrEq("GPU.1"),
                   StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(dgpuFullDeviceName));

       ON_CALL(*plugin, ParseMetaDevices).WillByDefault([this](const std::string& priorityDevices,
                   const std::map<std::string, std::string>& config) {
               return plugin->MultiDeviceInferencePlugin::ParseMetaDevices(priorityDevices, config);
               });
        std::tie(priorityDevices, metaDevices, throwException) = GetParam();
        sizeOfMetaDevices = static_cast<int>(metaDevices.size());
    }

    void compare(std::vector<DeviceInformation>& result, std::vector<DeviceInformation>& expect) {
        EXPECT_EQ(result.size(), expect.size());
        if (result.size() == expect.size()) {
            for (unsigned int i = 0 ; i < result.size(); i++) {
                EXPECT_EQ(result[i].deviceName, expect[i].deviceName);
                EXPECT_EQ(result[i].uniqueName, expect[i].uniqueName);
                EXPECT_EQ(result[i].numRequestsPerDevices, expect[i].numRequestsPerDevices);
                EXPECT_EQ(result[i].defaultDeviceID, expect[i].defaultDeviceID);
            }
        }
    }

    void compareDevicePriority(std::vector<DeviceInformation>& result, std::vector<DeviceInformation>& expect) {
        EXPECT_EQ(result.size(), expect.size());
        if (result.size() == expect.size()) {
            for (unsigned int i = 0 ; i < result.size(); i++) {
                EXPECT_EQ(result[i].devicePriority, expect[i].devicePriority);
            }
        }
    }

protected:
    // get Parameter
    std::string priorityDevices;
    std::vector<DeviceInformation> metaDevices;
    bool throwException;
    int sizeOfMetaDevices;
};
using ParseMetaDeviceNoIDTest = ParseMetaDeviceTest;

TEST_P(ParseMetaDeviceTest, ParseMetaDevicesWithPriority) {
    ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));
    IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, configKeys, {});
    ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
           .WillByDefault(RETURN_MOCK_VALUE(configKeys));

    EXPECT_CALL(*plugin, ParseMetaDevices(_, _)).Times(1);
    EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(*core, GetConfig(_, _)).Times(AnyNumber());
    EXPECT_CALL(*core, GetAvailableDevices()).Times(1);
    EXPECT_CALL(*core, GetSupportedConfig(_, _)).Times(sizeOfMetaDevices);
    if (throwException) {
        ASSERT_ANY_THROW(plugin->ParseMetaDevices(priorityDevices, {}));
    } else {
       auto result = plugin->ParseMetaDevices(priorityDevices, {{ov::device::priorities.name(), priorityDevices}});
       compare(result, metaDevices);
       compareDevicePriority(result, metaDevices);
    }
}

TEST_P(ParseMetaDeviceTest, ParseMetaDevicesNotWithPriority) {
    ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));
    IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, configKeys, {});
    ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
           .WillByDefault(RETURN_MOCK_VALUE(configKeys));

    EXPECT_CALL(*plugin, ParseMetaDevices(_, _)).Times(1 + !throwException);
    EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(*core, GetConfig(_, _)).Times(AnyNumber());
    EXPECT_CALL(*core, GetAvailableDevices()).Times(1 + !throwException);
    if (throwException) {
        ASSERT_ANY_THROW(plugin->ParseMetaDevices(priorityDevices, {}));
    } else {
       auto result = plugin->ParseMetaDevices(priorityDevices, {{}});
       compare(result, metaDevices);
       for (unsigned int i = 0 ; i < result.size(); i++) {
           EXPECT_EQ(result[i].devicePriority, 0);
       }
       auto result2 = plugin->ParseMetaDevices(priorityDevices, {{ov::device::priorities.name(), ""}});
       compare(result2, metaDevices);
       for (unsigned int i = 0 ; i < result.size(); i++) {
           EXPECT_EQ(result2[i].devicePriority, 0);
       }
    }
}

TEST_P(ParseMetaDeviceNoIDTest, ParseMetaDevices) {
    ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevsNoID));
    IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, configKeys, {CONFIG_KEY(DEVICE_ID)});
    ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
           .WillByDefault(RETURN_MOCK_VALUE(configKeys));
    EXPECT_CALL(*plugin, ParseMetaDevices(_, _)).Times(1);
    EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(*core, GetConfig(_, _)).Times(AnyNumber());
    EXPECT_CALL(*core, GetAvailableDevices()).Times(1);
    EXPECT_CALL(*core, GetSupportedConfig(_, _)).Times(sizeOfMetaDevices);
    if (throwException) {
        ASSERT_ANY_THROW(plugin->ParseMetaDevices(priorityDevices, {}));
    } else {
       auto result = plugin->ParseMetaDevices(priorityDevices, {{ov::device::priorities.name(), priorityDevices}});
       compare(result, metaDevices);
       compareDevicePriority(result, metaDevices);
    }
}
// ConfigParams details
// example
// ConfigParams {devicePriority, expect metaDevices, ifThrowException}

const std::vector<ConfigParams> testConfigs = {
    // ConfigParams {"CPU,GPU,MYRIAD,VPUX",
    //     {{"CPU", {}, -1, "", "CPU_", 0},
    //         {"GPU.0", {}, -1, "0", std::string(igpuFullDeviceName) + "_0", 1},
    //         {"GPU.1", {}, -1, "1", std::string(dgpuFullDeviceName) + "_1", 1},
    //         {"MYRIAD.9.2-ma2480", {}, -1, "9.2-ma2480", "MYRIAD_9.2-ma2480", 2},
    //         {"MYRIAD.9.1-ma2480", {}, -1, "9.1-ma2480", "MYRIAD_9.1-ma2480", 2},
    //         {"VPUX", {}, -1, "", "VPUX_", 3}}, false},
    // ConfigParams {"VPUX,MYRIAD,GPU,CPU",
    //     {{"VPUX", {}, -1, "", "VPUX_", 0},
    //         {"MYRIAD.9.2-ma2480", {}, -1, "9.2-ma2480", "MYRIAD_9.2-ma2480", 1},
    //         {"MYRIAD.9.1-ma2480", {}, -1, "9.1-ma2480", "MYRIAD_9.1-ma2480", 1},
    //         {"GPU.0", {}, -1, "0", std::string(igpuFullDeviceName) + "_0", 2},
    //         {"GPU.1", {}, -1, "1", std::string(dgpuFullDeviceName) + "_1", 2},
    //         {"CPU", {}, -1, "", "CPU_", 3}}, false},
    // ConfigParams {"CPU(1),GPU(2),MYRIAD(3),VPUX(4)",
    //     {{"CPU", {}, 1, "", "CPU_", 0},
    //         {"GPU.0", {}, 2, "0", std::string(igpuFullDeviceName) + "_0", 1},
    //         {"GPU.1", {}, 2, "1", std::string(dgpuFullDeviceName) + "_1", 1},
    //         {"MYRIAD.9.2-ma2480", {}, 3, "9.2-ma2480", "MYRIAD_9.2-ma2480", 2},
    //         {"MYRIAD.9.1-ma2480", {}, 3, "9.1-ma2480", "MYRIAD_9.1-ma2480", 2},
    //         {"VPUX", {}, 4, "", "VPUX_", 3}}, false},
    //
    ConfigParams {"CPU,GPU,VPUX",
         {{"CPU", {}, -1, "", "CPU_", 0},
             {"GPU.0", {}, -1, "", std::string(igpuFullDeviceName) + "_0", 1},
             {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1},
             {"VPUX", {}, -1, "", "VPUX_", 2}}, false},
    ConfigParams {"VPUX,GPU,CPU",
         {{"VPUX", {}, -1, "", "VPUX_", 0},
             {"GPU.0", {}, -1, "", std::string(igpuFullDeviceName) + "_0", 1},
             {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1},
             {"CPU", {}, -1, "", "CPU_", 2}}, false},
    ConfigParams {"CPU(1),GPU(2),VPUX(4)",
         {{"CPU", {}, 1, "", "CPU_", 0},
             {"GPU.0", {}, 2, "", std::string(igpuFullDeviceName) + "_0", 1},
             {"GPU.1", {}, 2, "", std::string(dgpuFullDeviceName) + "_1", 1},
             {"VPUX", {}, 4, "", "VPUX_", 2}}, false},

    ConfigParams {"CPU(-1),GPU,VPUX",  {}, true},
    ConfigParams {"CPU(NA),GPU,VPUX",  {}, true},

    ConfigParams {"CPU(3),GPU.1,VPUX",
        {{"CPU", {}, 3, "",  "CPU_", 0},
            {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1},
            {"VPUX", {}, -1, "", "VPUX_", 2}}, false},
    ConfigParams {"VPUX,GPU.1,CPU(3)",
        {{"VPUX", {}, -1, "", "VPUX_", 0},
            {"GPU.1", {}, -1, "", std::string(dgpuFullDeviceName) + "_1", 1},
            {"CPU", {}, 3, "",  "CPU_", 2}}, false}
};

const std::vector<ConfigParams> testConfigsNoID = {
    ConfigParams {"CPU,GPU,VPUX",
        {{"CPU", {}, -1, "", "CPU_", 0},
        {"GPU", {}, -1, "0", std::string(igpuFullDeviceName) + "_0", 1},
        {"VPUX", {}, -1, "", "VPUX_", 2}}, false},
};


INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ParseMetaDeviceTest,
                ::testing::ValuesIn(testConfigs),
            ParseMetaDeviceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ParseMetaDeviceNoIDTest,
                ::testing::ValuesIn(testConfigsNoID),
            ParseMetaDeviceTest::getTestCaseName);

//toDo need add test for ParseMetaDevices(_, config) to check device config of
//return metaDevices

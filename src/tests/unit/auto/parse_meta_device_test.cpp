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

const char cpuFullDeviceName[] = "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz";
const char igpuFullDeviceName[] = "Intel(R) Gen9 HD Graphics (iGPU)";
// const char dgpuFullDeviceName[] = "Intel(R) Iris(R) Xe MAX Graphics (dGPU)";
const char myriadFullDeviceName[] = "Intel Movidius Myriad X VPU";
const char vpuxFullDeviceName[] = "";
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

       ON_CALL(*core, GetMetric(HasSubstr(CommonTestUtils::DEVICE_CPU),
                   StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(cpuFullDeviceName));
       ON_CALL(*core, GetMetric(HasSubstr(CommonTestUtils::DEVICE_GPU),
                   StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(igpuFullDeviceName));
       ON_CALL(*core, GetMetric(HasSubstr(CommonTestUtils::DEVICE_MYRIAD),
                   StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(myriadFullDeviceName));
       ON_CALL(*core, GetMetric(HasSubstr(CommonTestUtils::DEVICE_KEEMBAY),
                   StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(vpuxFullDeviceName));
       IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, otherConfigKeys, {CONFIG_KEY(DEVICE_ID)});
       IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, cpuConfigKeys, {});
       ON_CALL(*core, GetMetric(HasSubstr(CommonTestUtils::DEVICE_CPU),
                   StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(RETURN_MOCK_VALUE(cpuConfigKeys));
       ON_CALL(*core, GetMetric(Not(HasSubstr(CommonTestUtils::DEVICE_CPU)),
                   StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(RETURN_MOCK_VALUE(otherConfigKeys));
       ON_CALL(*core, GetConfig(_, StrEq(CONFIG_KEY(DEVICE_ID))))
           .WillByDefault(InvokeWithoutArgs([](){return "01";}));

       ON_CALL(*plugin, ParseMetaDevices).WillByDefault([this](const std::string& priorityDevices,
                   const std::map<std::string, std::string>& config) {
               return plugin->MultiDeviceInferencePlugin::ParseMetaDevices(priorityDevices, config);
               });
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
};

TEST_P(ParseMetaDeviceTest, ParseMetaDevices) {
    // get Parameter
    std::string priorityDevices;
    std::vector<DeviceInformation> metaDevices;
    bool throwException;
    std::tie(priorityDevices, metaDevices, throwException) = this->GetParam();

    EXPECT_CALL(*plugin, ParseMetaDevices(_, _)).Times(1);
    EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AnyNumber());
    EXPECT_CALL(*core, GetConfig(_, _)).Times(AnyNumber());
    if (throwException) {
        ASSERT_ANY_THROW(plugin->ParseMetaDevices(priorityDevices, {}));
    } else {
       auto result = plugin->ParseMetaDevices(priorityDevices, {});
       compare(result, metaDevices);
    }
}

// ConfigParams details
// example
// ConfigParams {devicePriority, expect metaDevices, ifThrowException}

const std::vector<ConfigParams> testConfigs = {
    ConfigParams {"CPU,GPU,MYRIAD,VPUX",
        {{"CPU", {}, -1, "", "CPU_"},
            {"GPU", {}, -1, "01", std::string(igpuFullDeviceName) + "_01"},
            {"MYRIAD", {}, -1, "01", "MYRIAD_01"},
            {"VPUX", {}, -1, "01", "VPUX_01"}}, false},
    ConfigParams {"CPU(1),GPU(2),MYRIAD(3),VPUX(4)",
        {{"CPU", {}, 1, "", "CPU_"},
            {"GPU", {}, 2, "01", std::string(igpuFullDeviceName) + "_01"},
            {"MYRIAD", {}, 3, "01", "MYRIAD_01"},
            {"VPUX", {}, 4, "01", "VPUX_01"}}, false},
    ConfigParams {"CPU(-1),GPU,MYRIAD,VPUX",  {}, true},
    ConfigParams {"CPU(NA),GPU,MYRIAD,VPUX",  {}, true},
    ConfigParams {"CPU.02(3),GPU.03,MYRIAD.04,VPUX.05",
        {{"CPU.02", {}, 3, "",  "CPU_02"},
            {"GPU.03", {}, -1, "", std::string(igpuFullDeviceName) + "_03"},
            {"MYRIAD.04", {}, -1, "", "MYRIAD_04"},
            {"VPUX.05", {}, -1, "", "VPUX_05"}}, false}
    };


INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ParseMetaDeviceTest,
                ::testing::ValuesIn(testConfigs),
            ParseMetaDeviceTest::getTestCaseName);

//toDo need add test for ParseMetaDevices(_, config) to check device config of
//return metaDevices

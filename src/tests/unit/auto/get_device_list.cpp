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
#include "plugin/mock_auto_device_plugin.hpp"
#include "mock_common.hpp"

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
const std::vector<std::string> availableDevs = {"CPU", "GPU.0", "GPU.1", "VPUX", "UNSUPPORTED_DEVICE"};
using ConfigParams = std::tuple<
        std::string,                        // Priority devices
        std::string                         // expect metaDevices
        >;
class GetDeviceListTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<MockICore>                      core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string priorityDevices;
        std::string metaDevices;
        std::tie(priorityDevices, metaDevices) = obj.param;
        std::ostringstream result;
        result << "priorityDevices_" << priorityDevices;
        result << "_expectedDevices" << metaDevices;
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


       ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));

       ON_CALL(*plugin, GetDeviceList).WillByDefault([this](
                   const std::map<std::string, std::string>& config) {
               return plugin->MultiDeviceInferencePlugin::GetDeviceList(config);
               });
    }
};

TEST_P(GetDeviceListTest, GetDeviceListTestWithExcludeList) {
    // get Parameter
    std::string priorityDevices;
    std::string metaDevices;
    std::tie(priorityDevices, metaDevices) = this->GetParam();

    //EXPECT_CALL(*plugin, GetDeviceList(_)).Times(1);
    EXPECT_CALL(*core, GetAvailableDevices()).Times(1);
    auto result = plugin->GetDeviceList({{ov::device::priorities.name(), priorityDevices}});
    EXPECT_EQ(result, metaDevices);
}


// ConfigParams details
// example
// ConfigParams {devicePriority, expect metaDevices, ifThrowException}

const std::vector<ConfigParams> testConfigs = {
    //
    ConfigParams {"CPU,GPU,VPUX",
        "CPU,GPU,VPUX"},
    ConfigParams {"VPUX,GPU,CPU,-GPU.0",
        "VPUX,GPU.1,CPU"},
    ConfigParams {"-GPU.0,GPU,CPU",
        "GPU.1,CPU"},
    ConfigParams {"-GPU.0,GPU",
        "GPU.1"},
    ConfigParams {"-GPU.0", "CPU,GPU.1,VPUX"},
    ConfigParams {"-GPU.0,-GPU.1", "CPU,VPUX"},
    ConfigParams {"-GPU.0,-CPU", "GPU.1,VPUX"}
};


INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, GetDeviceListTest,
                ::testing::ValuesIn(testConfigs),
            GetDeviceListTest::getTestCaseName);

//toDo need add test for ParseMetaDevices(_, config) to check device config of
//return metaDevices

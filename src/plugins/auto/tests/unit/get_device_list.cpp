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

using ::testing::Return;
using ::testing::Property;
using ::testing::Eq;
using ::testing::AnyNumber;
using ::testing::ReturnRef;
using ::testing::NiceMock;
using ::testing::AtLeast;
using ::testing::InvokeWithoutArgs;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

const std::vector<std::string> availableDevs = {"CPU", "GPU", "VPUX"};
const std::vector<std::string> availableDevsWithId = {"CPU", "GPU.0", "GPU.1", "VPUX"};
using Params = std::tuple<std::string, std::string>;
using ConfigParams = std::tuple<
        std::vector<std::string>,           // Available devices retrieved from Core
        Params                              // Params {devicePriority, expect metaDevices}
        >;
class GetDeviceListTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>> plugin;

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

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
       // prepare mockicore and cnnNetwork for loading
       core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
       auto* origin_plugin = new NiceMock<MockMultiDeviceInferencePlugin>();
       plugin = std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>>(origin_plugin);
       // replace core with mock Icore
       plugin->SetCore(core);

       ON_CALL(*plugin, GetDeviceList).WillByDefault([this](
                   const std::map<std::string, std::string>& config) {
               return plugin->MultiDeviceInferencePlugin::GetDeviceList(config);
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

    ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));
    EXPECT_CALL(*core, GetAvailableDevices()).Times(1);
    auto result = plugin->GetDeviceList({{ov::device::priorities.name(), priorityDevices}});
    EXPECT_EQ(result, metaDevices);
}

const std::vector<Params> testConfigsWithId = {Params{" ", " "},
                                         Params{"", "CPU,GPU.0,GPU.1"},
                                         Params{"CPU, ", "CPU, "},
                                         Params{" ,CPU", " ,CPU"},
                                         Params{"CPU,", "CPU"},
                                         Params{"CPU,,GPU", "CPU,GPU.0,GPU.1"},
                                         Params{"CPU, ,GPU", "CPU, ,GPU.0,GPU.1"},
                                         Params{"CPU,GPU,GPU.1", "CPU,GPU.0,GPU.1"},
                                         Params{"CPU,GPU,VPUX,INVALID_DEVICE", "CPU,GPU.0,GPU.1,VPUX,INVALID_DEVICE"},
                                         Params{"VPUX,GPU,CPU,-GPU.0", "VPUX,GPU.1,CPU"},
                                         Params{"-GPU.0,GPU,CPU", "GPU.1,CPU"},
                                         Params{"-GPU.0,GPU", "GPU.1"},
                                         Params{"-GPU,GPU.0", "GPU.0"},
                                         Params{"-GPU.0", "CPU,GPU.1"},
                                         Params{"-GPU.0,-GPU.1", "CPU"},
                                         Params{"-GPU.0,-GPU.1,INVALID_DEVICE", "INVALID_DEVICE"},
                                         Params{"-GPU.0,-GPU.1,-INVALID_DEVICE", "CPU"},
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
                                         Params{"CPU,VPUX", "CPU,VPUX"},
                                         Params{"CPU,-VPUX", "CPU"},
                                         Params{"CPU,-INVALID_DEVICE", "CPU"},
                                         Params{"CPU,GPU,VPUX", "CPU,GPU,VPUX"}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_GetDeviceListWithID,
                         GetDeviceListTest,
                         ::testing::Combine(::testing::Values(availableDevsWithId),
                                            ::testing::ValuesIn(testConfigsWithId)),
                         GetDeviceListTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_GetDeviceList,
                         GetDeviceListTest,
                         ::testing::Combine(::testing::Values(availableDevs), ::testing::ValuesIn(testConfigs)),
                         GetDeviceListTest::getTestCaseName);

//toDo need add test for ParseMetaDevices(_, config) to check device config of
//return metaDevices

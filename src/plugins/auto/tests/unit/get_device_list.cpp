// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"

using Config = std::map<std::string, std::string>;
using namespace ov::mock_auto_plugin;

const std::vector<std::string> availableDevs = {"CPU", "GPU", "NPU"};
const std::vector<std::string> availableDevsWithId = {"CPU", "GPU.0", "GPU.1", "NPU"};
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
        OV_ASSERT_NO_THROW(result = plugin->get_device_list({ov::device::priorities(priorityDevices)}));
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
        OV_ASSERT_NO_THROW(result = plugin->get_device_list({ov::device::priorities(priorityDevices)}));
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

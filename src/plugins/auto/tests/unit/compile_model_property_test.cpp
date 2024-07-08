// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "include/auto_unit_test.hpp"
#include "openvino/runtime/properties.hpp"

// define a matcher if all the elements of subMap are contained in the map.
MATCHER_P(MapContains, subMap, "Check if all the elements of the subMap are contained in the map.") {
    if (subMap.empty())
        return true;
    for (auto& item : subMap) {
        auto key = item.first;
        auto value = item.second;
        auto dest = arg.find(key);
        if (dest == arg.end()) {
            return false;
        } else if (dest->second != value) {
            return false;
        }
    }
    return true;
}
using namespace ov::mock_auto_plugin;

using ConfigParams = std::tuple<std::string,               // virtual device name to load network
                                std::vector<std::string>,  // hardware device name to expect loading network on
                                ov::AnyMap>;               // secondary property setting to device

static std::vector<ConfigParams> testConfigs;

class LoadNetworkWithSecondaryConfigsMockTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string deviceName;
        std::vector<std::string> targetDevices;
        ov::AnyMap deviceConfigs;
        std::tie(deviceName, targetDevices, deviceConfigs) = obj.param;
        std::ostringstream result;
        result << "_virtual_device_" << deviceName;
        result << "_loadnetwork_to_device_";
        for (auto& device : targetDevices) {
            result << device << "_";
        }
        for (auto& item : deviceConfigs) {
            result << item.first << "_" << item.second.as<std::string>() << "_";
        }
        auto name = result.str();
        name.pop_back();
        return name;
    }

    static std::vector<ConfigParams> CreateConfigs() {
        testConfigs.clear();
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"NUM_STREAMS", "12"},
                                            {"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU", "GPU"},
                                           {{"NUM_STREAMS", "15"},
                                            {"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:3}}"},
                                            {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:CPU",
                         {"CPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:CPU,GPU",
                         {"CPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:GPU",
                         {"GPU"},
                         {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:5}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:GPU,CPU",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:5}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});

        testConfigs.push_back(
            ConfigParams{"MULTI:CPU",
                         {"CPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});
        testConfigs.push_back(
            ConfigParams{"MULTI:CPU,GPU",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"MULTI:GPU",
                         {"GPU"},
                         {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:5}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});
        testConfigs.push_back(
            ConfigParams{"MULTI:GPU,CPU",
                         {"CPU", "GPU"},
                         {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:5}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        return testConfigs;
    }

    void SetUp() override {
        std::vector<std::string> availableDevs = {"CPU", "GPU"};
        ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(Return(mockExeNetwork));
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrNe(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(Return(mockExeNetworkActual));
    }
};

TEST_P(LoadNetworkWithSecondaryConfigsMockTest, LoadNetworkWithSecondaryConfigsTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos)
        plugin->set_device_name("AUTO");
    if (device.find("MULTI") != std::string::npos)
        plugin->set_device_name("MULTI");

    for (auto& deviceName : targetDevices) {
        auto item = config.find(ov::device::properties.name());
        ov::AnyMap deviceConfigs;
        if (item != config.end()) {
            ov::AnyMap devicesProperties;
            std::stringstream strConfigs(item->second.as<std::string>());
            // Parse the device properties to common property into deviceConfigs.
            ov::util::Read<ov::AnyMap>{}(strConfigs, devicesProperties);
            auto it = devicesProperties.find(deviceName);
            if (it != devicesProperties.end()) {
                std::stringstream strConfigs(it->second.as<std::string>());
                ov::util::Read<ov::AnyMap>{}(strConfigs, deviceConfigs);
            }
        }
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(deviceName),
                                  ::testing::Matcher<const ov::AnyMap&>(MapContains(deviceConfigs))))
            .Times(1);
    }

    OV_ASSERT_NO_THROW(plugin->compile_model(model, config));
}

using AutoLoadExeNetworkFailedTest = LoadNetworkWithSecondaryConfigsMockTest;
TEST_P(AutoLoadExeNetworkFailedTest, checkLoadFailMassage) {
    std::string device;
    std::vector<std::string> targetDevices;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos)
        plugin->set_device_name("AUTO");
    if (device.find("MULTI") != std::string::npos)
        plugin->set_device_name("MULTI");

    const auto cpu_failed = std::string{"Mock CPU Load Failed"};
    const auto gpu_failed = std::string{"Mock GPU Load Failed"};
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                          ::testing::Matcher<const ov::AnyMap&>(_)))
        .WillByDefault(ov::Throw(gpu_failed));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                          ::testing::Matcher<const ov::AnyMap&>(_)))
        .WillByDefault(ov::Throw(cpu_failed));

    const auto auto_failed = std::string{"[AUTO] compile model failed"};
    const auto multi_failed = std::string{"[MULTI] compile model failed"};
    if (device == "AUTO") {
        OV_EXPECT_THROW(plugin->compile_model(model, config),
                        ov::Exception,
                        AllOf(HasSubstr(auto_failed), HasSubstr(cpu_failed), HasSubstr(gpu_failed)));
    } else if (device == "AUTO:CPU") {
        OV_EXPECT_THROW(plugin->compile_model(model, config),
                        ov::Exception,
                        AllOf(HasSubstr(auto_failed), HasSubstr(cpu_failed)));
    } else if (device == "AUTO:GPU") {
        OV_EXPECT_THROW(plugin->compile_model(model, config),
                        ov::Exception,
                        AllOf(HasSubstr(auto_failed), HasSubstr(gpu_failed)));
    } else if (device == "MULTI") {
        OV_EXPECT_THROW(plugin->compile_model(model, config),
                        ov::Exception,
                        AllOf(HasSubstr(multi_failed), HasSubstr(cpu_failed), HasSubstr(gpu_failed)));
    } else if (device == "MULTI:CPU") {
        OV_EXPECT_THROW(plugin->compile_model(model, config),
                        ov::Exception,
                        AllOf(HasSubstr(multi_failed), HasSubstr(cpu_failed)));
    } else if (device == "MULTI:GPU") {
        OV_EXPECT_THROW(plugin->compile_model(model, config),
                        ov::Exception,
                        AllOf(HasSubstr(multi_failed), HasSubstr(gpu_failed)));
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_AutoMock_LoadNetworkWithSecondaryConfigs,
                         LoadNetworkWithSecondaryConfigsMockTest,
                         ::testing::ValuesIn(LoadNetworkWithSecondaryConfigsMockTest::CreateConfigs()),
                         LoadNetworkWithSecondaryConfigsMockTest::getTestCaseName);

const std::vector<ConfigParams> testConfigsAutoLoadFailed = {
    ConfigParams{"AUTO", {"CPU", "GPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}},
    ConfigParams{"AUTO:CPU", {"CPU"}, {{"MULTI_DEVICE_PRIORITIES", "CPU"}}},
    ConfigParams{"AUTO:GPU", {"GPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU"}}},
    ConfigParams{"MULTI", {"CPU", "GPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}},
    ConfigParams{"MULTI:CPU", {"CPU"}, {{"MULTI_DEVICE_PRIORITIES", "CPU"}}},
    ConfigParams{"MULTI:GPU", {"GPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU"}}}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoLoadExeNetworkFailedTest,
                         AutoLoadExeNetworkFailedTest,
                         ::testing::ValuesIn(testConfigsAutoLoadFailed),
                         AutoLoadExeNetworkFailedTest::getTestCaseName);

using PropertyTestParams = std::tuple<std::string,                  // virtual device name to load network
                                      std::string,                  // device priority
                                      std::map<std::string, bool>,  // if supported property
                                      ov::AnyMap>;                  // optional property and its expected value

class CompiledModelPropertyMockTest : public tests::AutoTest, public ::testing::TestWithParam<PropertyTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertyTestParams> obj) {
        std::string deviceName;
        std::string devicePriorities;
        std::map<std::string, bool> isSupportProperty;
        ov::AnyMap properties;
        std::tie(deviceName, devicePriorities, isSupportProperty, properties) = obj.param;
        std::ostringstream result;
        result << "_virtual_device_" << deviceName;
        result << "_loadnetwork_to_device_" << devicePriorities;
        for (auto& property : properties) {
            result << "_property_" << property.first;
            bool isSupport = isSupportProperty[property.first];
            if (isSupport)
                result << "_isSupport_No_";
            else
                result << "_isSupport_Yes_";
            result << "_expectedValue_" << property.second.as<std::string>();
        }
        return result.str();
    }

    void SetUp() override {
        std::string deviceName;
        std::string devicePriorities;
        ov::AnyMap properties;
        std::map<std::string, bool> isSupportProperty;
        std::tie(deviceName, devicePriorities, isSupportProperty, properties) = GetParam();
        std::vector<std::string> availableDevs = {"CPU", "GPU"};
        ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(Return(mockExeNetwork));
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrNe(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(Return(mockExeNetworkActual));
        std::vector<ov::PropertyName> supported_props = {};
        for (auto& property : properties) {
            bool isSupport = isSupportProperty[property.first];
            if (isSupport) {
                supported_props.push_back(property.first);
                auto value = property.second.as<std::string>();
                ON_CALL(*mockIExeNet.get(), get_property(StrEq(property.first)))
                    .WillByDefault(RETURN_MOCK_VALUE(value));
                ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(property.first)))
                    .WillByDefault(RETURN_MOCK_VALUE(value));
            } else {
                ON_CALL(*mockIExeNet.get(), get_property(StrEq(property.first)))
                    .WillByDefault(ov::Throw("unsupported property"));
                ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(property.first)))
                    .WillByDefault(ov::Throw("unsupported property"));
            }
        }
        ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::supported_properties.name())))
            .WillByDefault(Return(ov::Any(supported_props)));
        ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::supported_properties.name())))
            .WillByDefault(Return(ov::Any(supported_props)));
    }
};

TEST_P(CompiledModelPropertyMockTest, compiledModelGetPropertyNoThrow) {
    std::string deviceName;
    std::string devicePriorities;
    ov::AnyMap properties;
    std::map<std::string, bool> isSupportProperty;
    std::tie(deviceName, devicePriorities, isSupportProperty, properties) = GetParam();
    if (deviceName.find("AUTO") != std::string::npos)
        plugin->set_device_name("AUTO");
    if (deviceName.find("MULTI") != std::string::npos)
        plugin->set_device_name("MULTI");
    std::shared_ptr<ov::ICompiledModel> autoExecNetwork;
    OV_ASSERT_NO_THROW(autoExecNetwork = plugin->compile_model(model, {ov::device::priorities(devicePriorities)}));
    for (auto& property : properties) {
        auto result = autoExecNetwork->get_property(property.first).as<std::string>();
        EXPECT_EQ(result, property.second.as<std::string>());
    }
}
const std::vector<PropertyTestParams> testCompiledModelProperty = {
    PropertyTestParams{"AUTO",
                       "CPU,GPU",
                       {{ov::loaded_from_cache.name(), true}},
                       {{ov::loaded_from_cache.name(), true}}},
    PropertyTestParams{"AUTO",
                       "CPU,GPU",
                       {{ov::loaded_from_cache.name(), true}},
                       {{ov::loaded_from_cache.name(), false}}},
    PropertyTestParams{"AUTO",
                       "CPU,GPU",
                       {{ov::loaded_from_cache.name(), false}},
                       {{ov::loaded_from_cache.name(), false}}},
    PropertyTestParams{"MULTI",
                       "CPU,GPU",
                       {{ov::loaded_from_cache.name(), true}},
                       {{ov::loaded_from_cache.name(), true}}},
    PropertyTestParams{"MULTI",
                       "CPU,GPU",
                       {{ov::loaded_from_cache.name(), true}},
                       {{ov::loaded_from_cache.name(), false}}},
    PropertyTestParams{"MULTI",
                       "CPU,GPU",
                       {{ov::loaded_from_cache.name(), false}},
                       {{ov::loaded_from_cache.name(), false}}}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoCompiledModelPropertyMockTest,
                         CompiledModelPropertyMockTest,
                         ::testing::ValuesIn(testCompiledModelProperty),
                         CompiledModelPropertyMockTest::getTestCaseName);

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"

#define EXPECT_THROW_WITH_MESSAGE(stmt, etype, whatstring) EXPECT_THROW( \
        try { \
            stmt; \
        } catch (const etype& ex) { \
            EXPECT_THAT(std::string(ex.what()), HasSubstr(whatstring)); \
            throw; \
        } \
    , etype)
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
                                ov::AnyMap>;                   // secondary property setting to device

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
            ConfigParams{"AUTO", {"CPU"}, {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO", {"CPU", "GPU"}, {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:CPU", {"CPU"}, {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:CPU,GPU", {"CPU"}, {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:GPU", {"GPU"}, {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:5}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});
        testConfigs.push_back(ConfigParams{"AUTO:GPU,CPU",
                                           {"CPU", "GPU"},
                                           {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:5}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});

        testConfigs.push_back(
            ConfigParams{"MULTI:CPU", {"CPU"}, {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});
        testConfigs.push_back(ConfigParams{"MULTI:CPU,GPU",
                                           {"CPU", "GPU"},
                                           {{"DEVICE_PROPERTIES", "{CPU:{NUM_STREAMS:3}}"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"MULTI:GPU", {"GPU"}, {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:5}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});
        testConfigs.push_back(ConfigParams{"MULTI:GPU,CPU",
                                           {"CPU", "GPU"},
                                           {{"DEVICE_PROPERTIES", "{GPU:{NUM_STREAMS:5}}"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        return testConfigs;
    }

    void SetUp() override {
        std::vector<std::string> availableDevs = {"CPU", "GPU", "VPUX"};
        ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));
        ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
            ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)), _))
            .WillByDefault(Return(mockExeNetwork));
        ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                    ::testing::Matcher<const std::string&>(StrNe(CommonTestUtils::DEVICE_CPU)), _))
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
        EXPECT_CALL(
            *core,
           compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                        ::testing::Matcher<const std::string&>(deviceName),
                        ::testing::Matcher<const ov::AnyMap&>(MapContains(deviceConfigs))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->compile_model(model, config));
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

    ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const ov::AnyMap&>(_)))
                .WillByDefault(Throw(ov::Exception{"Mock GPU Load Failed"}));
    ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const ov::AnyMap&>(_)))
                .WillByDefault(Throw(ov::Exception{"Mock CPU Load Failed"}));
    if (device == "AUTO") {
        EXPECT_THROW_WITH_MESSAGE(plugin->compile_model(model, config), ov::Exception,
                                "[AUTO] compile model failed, GPU:Mock GPU Load Failed; CPU:Mock CPU Load Failed");
    } else if (device == "AUTO:CPU") {
        EXPECT_THROW_WITH_MESSAGE(plugin->compile_model(model, config), ov::Exception,
                                "[AUTO] compile model failed, CPU:Mock CPU Load Failed");
    } else if (device == "AUTO:GPU") {
        EXPECT_THROW_WITH_MESSAGE(plugin->compile_model(model, config), ov::Exception,
                                "[AUTO] compile model failed, GPU:Mock GPU Load Failed");
    } else if (device == "MULTI") {
        EXPECT_THROW_WITH_MESSAGE(plugin->compile_model(model, config), ov::Exception,
                                "[MULTI] compile model failed, GPU:Mock GPU Load Failed; CPU:Mock CPU Load Failed");
    } else if (device == "MULTI:CPU") {
        EXPECT_THROW_WITH_MESSAGE(plugin->compile_model(model, config), ov::Exception,
                                "[MULTI] compile model failed, CPU:Mock CPU Load Failed");
    } else if (device == "MULTI:GPU") {
        EXPECT_THROW_WITH_MESSAGE(plugin->compile_model(model, config), ov::Exception,
                                "[MULTI] compile model failed, GPU:Mock GPU Load Failed");
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
    ConfigParams{"MULTI:GPU", {"GPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU"}}}
    };

INSTANTIATE_TEST_SUITE_P(smoke_AutoLoadExeNetworkFailedTest, AutoLoadExeNetworkFailedTest,
                ::testing::ValuesIn(testConfigsAutoLoadFailed),
            AutoLoadExeNetworkFailedTest::getTestCaseName);

using AutoLimitCompilationThreadsForAcceleratorTest = LoadNetworkWithSecondaryConfigsMockTest;
TEST_P(AutoLimitCompilationThreadsForAcceleratorTest, checkPropertyCompileThreadsLimitationSetting) {
    std::string device;
    std::vector<std::string> targetDevices;
    std::tie(device, targetDevices, config) = this->GetParam();
    auto is_set_compilation = config.find(ov::compilation_num_threads.name());
    if (device.find("AUTO") != std::string::npos)
        plugin->set_device_name("AUTO");
    if (device.find("MULTI") != std::string::npos)
        plugin->set_device_name("MULTI");
    std::string actualSelectedDevice = targetDevices.front();
    std::string acceleratorDevice = targetDevices.back();
    int max_num_threads = -1;
    if (is_set_compilation != config.end()) {
        max_num_threads = is_set_compilation->second.as<int>();
    } else {
        const int num_logic_cores = ov::get_number_of_logical_cpu_cores();
        max_num_threads = (num_logic_cores / 2) == 0 ? 1 : (num_logic_cores / 2);
    }
    ov::AnyMap deviceConfigs = {ov::compilation_num_threads(max_num_threads)};

    std::vector<ov::PropertyName> supported_props = {ov::compilation_num_threads};
    ON_CALL(*core, get_supported_property).WillByDefault([](const std::string& device, const ov::AnyMap& fullConfigs) {
        // auto item = fullConfigs.find(device);
        ov::AnyMap deviceConfigs;
        for (auto&& item : fullConfigs) {
            if (item.first.find(device) != std::string::npos) {
                std::stringstream strConfigs(item.second.as<std::string>());
                ov::util::Read<ov::AnyMap>{}(strConfigs, deviceConfigs);
            } else if (item.first == ov::compilation_num_threads.name()) {
                deviceConfigs.insert(item);
            }
        }
        return deviceConfigs;
    });
    ON_CALL(*core, get_property(StrEq(actualSelectedDevice), StrEq(ov::supported_properties.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(supported_props));

    // set default value of ov::compilation_num_threads for actual selected device
    // set std::thread::hardware_concurrency() to GPU as default value
    int32_t test_value = static_cast<int>(std::thread::hardware_concurrency());
    ON_CALL(*core, get_property(StrEq(actualSelectedDevice), StrEq(ov::compilation_num_threads.name()), _))
        .WillByDefault(Return(test_value));
    // set 0 to other devices as the default value
    ON_CALL(*core, get_property(StrNe(actualSelectedDevice), StrEq(ov::compilation_num_threads.name()), _))
        .WillByDefault(Return((int32_t)0));

    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(acceleratorDevice)),
                              ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(1);

    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(actualSelectedDevice)),
                              ::testing::Matcher<const ov::AnyMap&>(MapContains(deviceConfigs))))
        .Times(1);

    ASSERT_NO_THROW(plugin->compile_model(model, config));
}
const std::vector<ConfigParams> testConfigsCompilationThreads = {
    ConfigParams{"AUTO", {"GPU", "CPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}},
    ConfigParams{"AUTO", {"GPU", "CPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}, ov::compilation_num_threads(2)}},
    ConfigParams{"AUTO", {"VPUX", "CPU"}, {{"MULTI_DEVICE_PRIORITIES", "VPUX,CPU"}}},
    ConfigParams{"AUTO", {"VPUX", "CPU"}, {{"MULTI_DEVICE_PRIORITIES", "VPUX,CPU"}, ov::compilation_num_threads(2)}}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMock_LimitCompileThreadsForAcceleratorTest,
                         AutoLimitCompilationThreadsForAcceleratorTest,
                         ::testing::ValuesIn(testConfigsCompilationThreads),
                         AutoLimitCompilationThreadsForAcceleratorTest::getTestCaseName);

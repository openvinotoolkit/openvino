// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include <thread>

#include "include/auto_unit_test.hpp"

using Config = std::map<std::string, std::string>;
using namespace ov::mock_auto_plugin;

using ConfigParams = std::tuple<bool,  // cpu load success
                                bool   // hw device load success
                                >;
class AutoReleaseHelperTest : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        bool cpuSuccess;
        bool accSuccess;
        std::tie(cpuSuccess, accSuccess) = obj.param;
        std::ostringstream result;
        if (!cpuSuccess) {
            result << "cpuLoadFailure_";
        } else {
            result << "cpuLoadSuccess_";
        }
        if (!accSuccess) {
            result << "accelerateorLoadFailure";
        } else {
            result << "accelerateorLoadSuccess";
        }
        return result.str();
    }
};

using PropertyConfigParams = std::tuple<bool,  // if CPU device sleep
                                        bool   // if actual device sleep
                                        >;
class AutoCompiledModelGetPropertyWithReleaseHelper : public tests::AutoTest,
                                                      public ::testing::TestWithParam<PropertyConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertyConfigParams> obj) {
        bool cpuSleep;
        bool actSleep;
        std::tie(cpuSleep, actSleep) = obj.param;
        std::ostringstream result;
        if (cpuSleep) {
            result << "cpuHelper_sleep_Yes_";
        } else {
            result << "cpuHelper_sleep_No_";
        }

        if (actSleep) {
            result << "actDevice_sleep_Yes_";
        } else {
            result << "actDevice_sleep_No_";
        }

        return result.str();
    }
};

TEST_P(AutoCompiledModelGetPropertyWithReleaseHelper, getPropertyTestAfterReleaseResourceNoThrow) {
    // get Parameter
    bool cpuSleep;
    bool actSleep;
    std::tie(cpuSleep, actSleep) = this->GetParam();
    // test auto plugin
    plugin->set_device_name("AUTO");
    const std::string strDevices = ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU;
    std::string modelNameCpu = "Text model name for CPU";
    std::string modelNameActual = "Text model name for actual device";

    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::model_name.name())))
        .WillByDefault(Return(ov::Any(modelNameCpu)));
    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::model_name.name())))
        .WillByDefault(Return(ov::Any(modelNameActual)));

    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::loaded_from_cache.name())))
        .WillByDefault(Return(ov::Any(false)));
    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::loaded_from_cache.name())))
        .WillByDefault(Return(ov::Any(false)));

    if (actSleep) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                return mockExeNetworkActual;
            }));
    } else {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                return mockExeNetworkActual;
            }));
    }
    if (cpuSleep) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                return mockExeNetwork;
            }));
    } else {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                return mockExeNetwork;
            }));
    }
    metaDevices = {{ov::test::utils::DEVICE_CPU, {}, -1}, {ov::test::utils::DEVICE_GPU, {}, -1}};
    DeviceInformation devInfo;
    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, get_valid_device)
        .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    ON_CALL(*plugin, select_device(Property(&std::vector<DeviceInformation>::size, Eq(2)), _, _))
        .WillByDefault(Return(metaDevices[1]));
    ON_CALL(*plugin, select_device(Property(&std::vector<DeviceInformation>::size, Eq(1)), _, _))
        .WillByDefault(Return(metaDevices[0]));
    config.insert(ov::device::priorities(ov::test::utils::DEVICE_CPU + std::string(",") + ov::test::utils::DEVICE_GPU));
    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    std::string result;
    OV_ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));
    if (actSleep) {
        if (!cpuSleep) {
            OV_ASSERT_NO_THROW(result = exeNetwork->get_property(ov::model_name.name()).as<std::string>());
            EXPECT_EQ(result, modelNameCpu);
        }
    } else {
        OV_ASSERT_NO_THROW(result = exeNetwork->get_property(ov::model_name.name()).as<std::string>());
    }

    auto supported_config_keys =
        exeNetwork->get_property(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
    for (const auto& cfg : supported_config_keys) {
        if (cfg == ov::model_name)
            continue;
        OV_ASSERT_NO_THROW(exeNetwork->get_property(cfg).as<std::string>());
    }
}

TEST_P(AutoReleaseHelperTest, releaseResource) {
    // get Parameter
    bool cpuSuccess;
    bool accSuccess;
    std::tie(cpuSuccess, accSuccess) = this->GetParam();
    size_t decreaseCount = 0;
    // test auto plugin
    plugin->set_device_name("AUTO");
    const std::string strDevices = ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU;

    if (accSuccess) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                return mockExeNetworkActual;
            }));
    } else {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                OPENVINO_THROW("");
                return mockExeNetworkActual;
            }));
    }
    if (cpuSuccess) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(Return(mockExeNetwork));
        if (accSuccess)
            decreaseCount++;
    } else {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              _))
            .WillByDefault(ov::Throw("GeneralError"));
    }
    metaDevices = {{ov::test::utils::DEVICE_CPU, {}, -1}, {ov::test::utils::DEVICE_GPU, {}, -1}};
    DeviceInformation devInfo;
    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, get_valid_device)
        .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    ON_CALL(*plugin, select_device(Property(&std::vector<DeviceInformation>::size, Eq(2)), _, _))
        .WillByDefault(Return(metaDevices[1]));
    ON_CALL(*plugin, select_device(Property(&std::vector<DeviceInformation>::size, Eq(1)), _, _))
        .WillByDefault(Return(metaDevices[0]));
    config.insert(ov::device::priorities(ov::test::utils::DEVICE_CPU + std::string(",") + ov::test::utils::DEVICE_GPU));
    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    if (cpuSuccess || accSuccess) {
        OV_ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));
        if (!cpuSuccess)
            EXPECT_EQ(exeNetwork->get_property(ov::execution_devices.name()).as<std::string>(),
                      ov::test::utils::DEVICE_GPU);
        else
            EXPECT_EQ(exeNetwork->get_property(ov::execution_devices.name()).as<std::string>(), "(CPU)");
    } else {
        ASSERT_THROW(exeNetwork = plugin->compile_model(model, config), ov::Exception);
    }
    auto sharedcount = mockExeNetwork._ptr.use_count();
    auto requestsharedcount = inferReqInternal.use_count();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_EQ(mockExeNetwork._ptr.use_count(), sharedcount - decreaseCount);
    EXPECT_EQ(inferReqInternal.use_count(), requestsharedcount - decreaseCount);
    if (cpuSuccess || accSuccess) {
        if (accSuccess)
            EXPECT_EQ(exeNetwork->get_property(ov::execution_devices.name()).as<std::string>(),
                      ov::test::utils::DEVICE_GPU);
        else
            EXPECT_EQ(exeNetwork->get_property(ov::execution_devices.name()).as<std::string>(),
                      ov::test::utils::DEVICE_CPU);
    }
}

//
const std::vector<ConfigParams> testConfigs = {ConfigParams{true, true},
                                               ConfigParams{true, false},
                                               ConfigParams{false, true},
                                               ConfigParams{false, false}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         AutoReleaseHelperTest,
                         ::testing::ValuesIn(testConfigs),
                         AutoReleaseHelperTest::getTestCaseName);

const std::vector<ConfigParams> testReleaseConfigs = {ConfigParams{false, true},
                                                      ConfigParams{false, false},
                                                      ConfigParams{true, false}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         AutoCompiledModelGetPropertyWithReleaseHelper,
                         ::testing::ValuesIn(testReleaseConfigs),
                         AutoCompiledModelGetPropertyWithReleaseHelper::getTestCaseName);

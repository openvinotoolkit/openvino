// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "include/auto_unit_test.hpp"

using namespace ov::mock_auto_plugin;

using ConfigParams = std::tuple<bool, ov::AnyMap>;

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
        } else {
            ov::Any val = dest->second;
            if (val.as<std::string>() != value) {
                return false;
            }
        }
    }
    return true;
}

class AutoStartupFallback : public tests::AutoTest, public ::testing::TestWithParam<ConfigParams> {
public:
    void SetUp() override {
        plugin->set_device_name("AUTO");
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(_),
                              _))
            .WillByDefault(Return(mockExeNetwork));
        metaDevices = {{ov::test::utils::DEVICE_CPU, {}, -1}, {ov::test::utils::DEVICE_GPU, {}, -1}};
        ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });
        ON_CALL(*plugin, select_device(_, _, _)).WillByDefault(Return(metaDevices[1]));
    }
};

TEST_P(AutoStartupFallback, propertytest) {
    // get Parameter
    bool startup_fallback;
    ov::AnyMap config;
    std::tie(startup_fallback, config) = this->GetParam();

    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(ov::test::utils::DEVICE_GPU),
                              _))
        .Times(1);
    if (startup_fallback) {
        std::map<std::string, std::string> test_map = {{"PERFORMANCE_HINT", "LATENCY"}};
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(ov::test::utils::DEVICE_CPU),
                                  ::testing::Matcher<const ov::AnyMap&>(MapContains(test_map))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->compile_model(model, config));
}

const std::vector<ConfigParams> testConfigs = {ConfigParams{true, {{"ENABLE_STARTUP_FALLBACK", "YES"}}},
                                               ConfigParams{false, {{"ENABLE_STARTUP_FALLBACK", "NO"}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_StartupFallback, AutoStartupFallback, ::testing::ValuesIn(testConfigs));

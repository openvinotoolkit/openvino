// Copyright (C) 2018-2026 Intel Corporation
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
    static std::string getTestCaseNameCacheTest(testing::TestParamInfo<ConfigParams> obj) {
        const auto& [startup_fallback, config] = obj.param;
        std::ostringstream result;
        result << "_expected_disabling_cache_" << startup_fallback;
        result << "_compiled_config_";
        for (auto& item : config) {
            result << item.first << "_" << item.second.as<std::string>() << "_";
        }
        auto name = result.str();
        name.pop_back();
        return name;
    }
    void SetUp() override {
        plugin->set_device_name("AUTO");
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(_),
                              _))
            .WillByDefault(Return(mockExeNetwork));
        metaDevices = {{ov::test::utils::DEVICE_CPU, {ov::cache_dir("test_dir")}, -1},
                       {ov::test::utils::DEVICE_GPU, {ov::cache_dir("test_dir")}, -1}};
        ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });
        ON_CALL(*plugin, select_device(_, _, _, _, _)).WillByDefault(Return(metaDevices[1]));
    }
};

TEST_P(AutoStartupFallback, propertytest) {
    // get Parameter

    const auto& [startup_fallback, config] = this->GetParam();

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

    OV_ASSERT_NO_THROW(plugin->compile_model(model, config));
}

const std::vector<ConfigParams> testConfigs = {ConfigParams{true, {{"ENABLE_STARTUP_FALLBACK", "YES"}}},
                                               ConfigParams{false, {{"ENABLE_STARTUP_FALLBACK", "NO"}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_StartupFallback, AutoStartupFallback, ::testing::ValuesIn(testConfigs));

using AutoLoadExeNetworkCacheDirSettingTest = AutoStartupFallback;
TEST_P(AutoLoadExeNetworkCacheDirSettingTest, canDisableCacheDirSettingForCPUPlugin) {
    // get Parameter

    const auto& [is_disable_cache_dir, config] = this->GetParam();
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                              _))
        .Times(1);

    if (is_disable_cache_dir) {
        std::map<std::string, std::string> test_map = {{ov::cache_dir.name(), ""}};
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                                  ::testing::Matcher<const ov::AnyMap&>(MapContains(test_map))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->compile_model(model, config));
}

const std::vector<ConfigParams> testCacheConfigs = {
    ConfigParams{true, {ov::intel_auto::enable_startup_fallback(true)}},
    ConfigParams{true, {ov::intel_auto::enable_runtime_fallback(true)}},
    ConfigParams{true, {ov::intel_auto::enable_startup_fallback(true), ov::intel_auto::enable_runtime_fallback(false)}},
    ConfigParams{false, {ov::intel_auto::enable_startup_fallback(false), ov::intel_auto::enable_runtime_fallback(true)}},
    ConfigParams{true, {ov::intel_auto::enable_startup_fallback(true), ov::intel_auto::enable_runtime_fallback(true)}},
    ConfigParams{false,
                 {ov::intel_auto::enable_startup_fallback(false), ov::intel_auto::enable_runtime_fallback(false)}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_disableCachingForCPUPlugin,
                         AutoLoadExeNetworkCacheDirSettingTest,
                         ::testing::ValuesIn(testCacheConfigs),
                         AutoLoadExeNetworkCacheDirSettingTest::getTestCaseNameCacheTest);

struct HelperCacheParams {
    ov::AnyMap actual_device_config;
    ov::AnyMap cpu_config;
    bool cpu_cache_disabled;
};

class AutoHelperCacheSetting : public tests::AutoTest, public ::testing::TestWithParam<HelperCacheParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<HelperCacheParams> obj) {
        const auto& [actual_config, cpu_config, disabled] = obj.param;
        std::string name = "actual_" + std::to_string(actual_config.size()) + "_cpu_";
        for (const auto& item : cpu_config) {
            name += item.first + "_";
        }
        return name + (disabled ? "disabled" : "untouched");
    }
    void SetUp() override {
        plugin->set_device_name("AUTO");
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(_),
                              _))
            .WillByDefault(Return(mockExeNetwork));
        const auto& [actual_config, cpu_config, disabled] = GetParam();
        metaDevices = {{ov::test::utils::DEVICE_CPU, cpu_config, -1},
                       {ov::test::utils::DEVICE_GPU, actual_config, -1}};
        ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
        ON_CALL(*plugin, get_valid_device)
            .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });
        ON_CALL(*plugin, select_device(_, _, _, _, _)).WillByDefault(Return(metaDevices[1]));
    }
};

TEST_P(AutoHelperCacheSetting, cpuHelperIsKeptOutOfTheCache) {
    const auto& [actual_config, cpu_config, disabled] = GetParam();
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                              _))
        .Times(1);
    // Core rejects a config that holds both properties, so the CPU helper must never get both
    const std::map<std::string, std::string> no_cache_dir = {{ov::cache_dir.name(), ""}};
    const auto cpu_matcher = disabled
                                 ? ::testing::Matcher<const ov::AnyMap&>(::testing::AllOf(
                                       MapContains(no_cache_dir),
                                       ::testing::Not(::testing::Contains(::testing::Key(ov::cache_path.name())))))
                                 : ::testing::Matcher<const ov::AnyMap&>(
                                       ::testing::Not(::testing::Contains(::testing::Key(ov::cache_dir.name()))));
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                              cpu_matcher))
        .Times(1);

    ASSERT_NO_THROW(plugin->compile_model(model, ov::AnyMap{ov::intel_auto::enable_startup_fallback(true)}));
}

const std::vector<HelperCacheParams> helperCacheConfigs = {
    // the cache of the CPU helper is disabled whichever property selects it
    HelperCacheParams{{ov::cache_dir("test_dir")}, {ov::cache_dir("test_dir")}, true},
    HelperCacheParams{{ov::cache_path(std::filesystem::path("test_cache.bin"))},
                      {ov::cache_path(std::filesystem::path("test_cache.bin"))},
                      true},
    HelperCacheParams{{}, {ov::cache_dir("test_dir")}, true},
    // nothing to disable when the CPU helper would not use a cache
    HelperCacheParams{{ov::cache_dir("test_dir")}, {}, false},
    HelperCacheParams{{}, {}, false}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_cpuHelperCacheSetting,
                         AutoHelperCacheSetting,
                         ::testing::ValuesIn(helperCacheConfigs),
                         AutoHelperCacheSetting::getTestCaseName);

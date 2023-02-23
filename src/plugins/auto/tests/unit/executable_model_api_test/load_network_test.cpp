// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_executable_network_test.hpp"

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
using namespace MockMultiDevice;

using LoadNetworkConfigParams = std::tuple<
                        std::string,                           // test MULTI or AUTO
                        std::map<ov::PropertyName, ov::Any>    // property to test
                        >;

class LoadNetworkMockTest : public LoadNetworkMockTestBase,
                            public ::testing::TestWithParam<LoadNetworkConfigParams> {
public:
    std::string pluginToTest;
    std::map<ov::PropertyName, ov::Any> properties;

public:
    static std::string getTestCaseName(testing::TestParamInfo<LoadNetworkConfigParams> obj) {
        std::string pluginname;
        std::map<ov::PropertyName, ov::Any> configuration;
        std::tie(pluginname, configuration) = obj.param;
        std::ostringstream result;
        result << pluginname;
        for (const std::pair<ov::PropertyName, ov::Any>& iter : configuration) {
            if (!iter.second.empty())
                result << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
            else
                result << "_" << iter.first << "_";
        }
        return result.str();
    }

    void TearDown() override {
    }

    void SetUp() override {
        std::tie(pluginToTest, properties) = GetParam();
        setupMock();
        // set device priority
        plugin->SetConfig({{ov::device::priorities.name(), "GPU,CPU"}});
        plugin->SetName(pluginToTest);
    }
};
using LoadNetworkInvalidMockTest = LoadNetworkMockTest;
TEST_P(LoadNetworkMockTest, LoadNetworkWithValidConfigsTest) {
    std::vector<std::string> configKeys = {"SUPPORTED_CONFIG_KEYS"};
    Config loadconfig;
    Config expectedconfig;
    for (const auto& property_item : properties) {
        configKeys.push_back(property_item.first);
    }
    ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(Return(configKeys));

    if (std::find(configKeys.begin(), configKeys.end(), ov::hint::performance_mode.name())  == configKeys.end()) {
        if (pluginToTest.find("AUTO") != std::string::npos)
            expectedconfig.insert({ov::hint::performance_mode.name(), "LATENCY"}); // default value for auto
        else
            expectedconfig.insert({ov::hint::performance_mode.name(), "THROUGHPUT"}); // default for multi
    }
    for (const auto& property_item : properties) {
        ov::Any default_value;
        if (property_item.first.is_mutable()) {
            loadconfig.insert({{property_item.first, property_item.second.as<std::string>()}});
            expectedconfig.insert({{property_item.first, property_item.second.as<std::string>()}});
            ON_CALL(*core, GetSupportedConfig(_, _)).WillByDefault(Return(expectedconfig));
            if (property_item.first.find(ov::device::priorities.name()) == std::string::npos
                && (property_item.first.find(ov::hint::performance_mode.name()) == std::string::npos && pluginToTest.find("AUTO") == std::string::npos)) {
                EXPECT_CALL(
                *core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(_),
                            ::testing::Matcher<const std::map<std::string, std::string>&>(
                                MapContains(expectedconfig))))
                .Times(2);
            }
            ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, loadconfig));
        }
    }
}

TEST_P(LoadNetworkInvalidMockTest, LoadNetworkWithValidConfigsTest) {
    for (const auto& property_item : properties) {
        ov::Any default_value;
        if (!property_item.second.empty()) {
            ASSERT_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, {{property_item.first, property_item.second.as<std::string>()}}), IE::Exception);
        }
    }
}
std::vector<std::string> test_load_targets = {"MULTI", "AUTO"};
INSTANTIATE_TEST_SUITE_P(smoke_Auto_LoadNetworkTests,
                         LoadNetworkMockTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(test_load_targets),
                            ::testing::ValuesIn((std::make_shared<ParamSet<TestParamType::VALID>>())->get_params())),
                         LoadNetworkMockTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_LoadNetworkTests,
                         LoadNetworkInvalidMockTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(test_load_targets),
                            ::testing::ValuesIn((std::make_shared<ParamSet<TestParamType::INVALID>>())->get_params())),
                         LoadNetworkInvalidMockTest::getTestCaseName);
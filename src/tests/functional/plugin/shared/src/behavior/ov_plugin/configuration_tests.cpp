// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/configuration_tests.hpp"
#include "openvino/runtime/properties.hpp"
#include <cstdint>

namespace BehaviorTestsDefinitions {

std::string OVDefaultConfigurationTest::getTestCaseName(const ::testing::TestParamInfo<DefaultConfigurationParameters> &obj) {
    std::string device_name;
    DefaultParameter default_parameter;
    std::tie(device_name, default_parameter) = obj.param;
    std::ostringstream result;
    result << "configKey=" << default_parameter._key << "_";
    result << "device_name=" << device_name;
    return result.str();
}

TEST_P(OVDefaultConfigurationTest, checkDeviceDefaultConfigurationValue) {
    device_name = std::get<DeviceName>(GetParam());
    std::string key;
    InferenceEngine::Parameter parameter;
    CustomComparator customComparator;
    default_parameter = std::get<DefaultParamterId>(GetParam());
    if (default_parameter._comparator) {
        auto expected = _core.get_property(device_name, default_parameter._key);
        auto &actual = parameter;
        ASSERT_TRUE(default_parameter._comparator(expected, actual)) << "For Key: " << default_parameter._key;
    } else {
        auto expected = _core->GetConfig(device_name, default_parameter._key).as<bool>();
        auto actual = default_parameter._parameter.as<bool>();
        ASSERT_EQ(expected, actual) <<
            "Expected: " << expected.as<std::string>() << "\n"
            "actual: " << actual.as<std::string>() <<;
    }
}

// Setting empty config doesn't throw
TEST_P(OVEmptyConfigTests, SetEmptyConfig) {
    AnyMap config;
    ASSERT_NO_THROW(core->get_property(device_name, ov::supported_properties));
    ASSERT_NO_THROW(core->SetConfig(device_name));
}

TEST_P(OVEmptyConfigTests, CanLoadNetworkWithEmptyConfig) {
    AnyMap config;
    ASSERT_NO_THROW(core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(core->LoadNetwork(cnnNet, device_name, config));
}

TEST_P(OVCorrectSingleOptionDefaultValueConfigTests, CheckDefaultValueOfConfig) {
    ASSERT_NO_THROW(core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_EQ(core->GetConfig(device_name, key), value);
}

// Setting correct config doesn't throw
TEST_P(OVCorrectConfigTests, SetCorrectConfig) {
    ASSERT_NO_THROW(core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(core->SetConfig(configuration, device_name));
}

TEST_P(OVCorrectConfigTests, CanLoadNetworkWithCorrectConfig) {
    ASSERT_NO_THROW(core->LoadNetwork(cnnNet, device_name, configuration));
}

TEST_P(OVCorrectConfigTests, CanUseCache) {
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    core->SetConfig({ { CONFIG_KEY(CACHE_DIR), "./test_cache" } });
    ASSERT_NO_THROW(core->LoadNetwork(cnnNet, device_name, configuration));
    ASSERT_NO_THROW(core->LoadNetwork(cnnNet, device_name, configuration));
    CommonTestUtils::removeDir("./test_cache");
}

TEST_P(OVCorrectConfigCheck, canSetConfigAndCheckGetConfig) {
    core->SetConfig(configuration, device_name);
    for (const auto& configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = core->GetConfig(device_name, configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(OVCorrectConfigCheck, canSetConfigTwiceAndCheckGetConfig) {
    core->SetConfig({}, device_name);
    core->SetConfig(configuration, device_name);
    for (const auto& configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = core->GetConfig(device_name, configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(OVCorrectSingleOptionCustomValueConfigTests, CheckCustomValueOfConfig) {
    ASSERT_NO_THROW(core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    AnyMap configuration = {{key, value}};
    ASSERT_NO_THROW(core->SetConfig(configuration, device_name));
    ASSERT_EQ(core->GetConfig(device_name, key), reference);
}

TEST_P(OVCorrectConfigPublicOptionsTests, CanSeePublicOption) {
    InferenceEngine::Parameter metric;
    ASSERT_NO_THROW(metric = core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    const auto& supportedOptions = metric.as<std::vector<std::string>>();
    ASSERT_NE(std::find(supportedOptions.cbegin(), supportedOptions.cend(), key), supportedOptions.cend());
}

TEST_P(OVCorrectConfigPrivateOptionsTests, CanNotSeePrivateOption) {
    InferenceEngine::Parameter metric;
    ASSERT_NO_THROW(metric = core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    const auto& supportedOptions = metric.as<std::vector<std::string>>();
    ASSERT_EQ(std::find(supportedOptions.cbegin(), supportedOptions.cend(), key), supportedOptions.cend());
}

TEST_P(OVIncorrectConfigTests, SetConfigWithIncorrectKey) {
    ASSERT_NO_THROW(core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_THROW(core->SetConfig(configuration, device_name), InferenceEngine::Exception);
}

TEST_P(OVIncorrectConfigTests, CanNotLoadNetworkWithIncorrectConfig) {
    ASSERT_THROW(auto execNet = core->LoadNetwork(cnnNet, device_name, configuration),
                 InferenceEngine::Exception);
}

TEST_P(OVIncorrectConfigSingleOptionTests, CanNotGetConfigWithIncorrectConfig) {
    ASSERT_NO_THROW(core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_THROW(core->GetConfig(device_name, key), InferenceEngine::Exception);
}

TEST_P(OVIncorrectConfigAPITests, SetConfigWithNoExistingKey) {
    ASSERT_NO_THROW(core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_THROW(core->SetConfig(configuration, device_name), InferenceEngine::Exception);
}


TEST_P(OVDefaultValuesConfigTests, CanSetDefaultValueBackToPlugin) {
    InferenceEngine::Parameter metric;
    ASSERT_NO_THROW(metric = core->GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> keys = metric;

    for (auto& key : keys) {
        InferenceEngine::Parameter configValue;
        ASSERT_NO_THROW(configValue = core->GetConfig(device_name, key));

        ASSERT_NO_THROW(core->SetConfig({{ key, configValue.as<std::string>()}}, device_name))
                                    << "device=" << device_name << " "
                                    << "config key=" << key << " "
                                    << "value=" << configValue.as<std::string>();
    }
}

} // namespace BehaviorTestsDefinitions

// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration_tests/configuration_tests.hpp"
#include <cstdint>

std::string DefaultConfigurationTest::getTestCaseName(const ::testing::TestParamInfo<DefaultConfigurationParameters> &obj) {
    std::string targetName;
    DefaultParameter defaultParameter;
    std::tie(targetName, defaultParameter) = obj.param;
    std::ostringstream result;
    result << "configKey=" << defaultParameter._key << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

TEST_P(DefaultConfigurationTest, checkDeviceDefaultConfigurationValue) {
    auto deviceName = std::get<DeviceName>(GetParam());
    std::string key;
    InferenceEngine::Parameter parameter;
    CustomComparator customComparator;
    auto defaultParameter = std::get<DefaultParamterId>(GetParam());
    if (defaultParameter._comparator) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key);
        auto& actual = parameter;
        ASSERT_TRUE(defaultParameter._comparator(expected, actual)) << "For Key: " << defaultParameter._key;
    } else if (defaultParameter._parameter.is<bool>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<bool>();
        auto actual = defaultParameter._parameter.as<bool>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<int>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<int>();
        auto actual = defaultParameter._parameter.as<int>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::uint32_t>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<std::uint32_t>();
        auto actual = defaultParameter._parameter.as<std::uint32_t>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<float>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<float>();
        auto actual = defaultParameter._parameter.as<float>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::string>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<std::string>();
        auto actual = defaultParameter._parameter.as<std::string>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::vector<std::string>>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<std::vector<std::string>>();
        auto actual = defaultParameter._parameter.as<std::vector<std::string>>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::vector<int>>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<std::vector<int>>();
        auto actual = defaultParameter._parameter.as<std::vector<int>>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::vector<std::uint32_t>>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<std::vector<std::uint32_t>>();
        auto actual = defaultParameter._parameter.as<std::vector<std::uint32_t>>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::vector<float>>()) {
        auto expected = _core->GetConfig(deviceName, defaultParameter._key).as<std::vector<float>>();
        auto actual = defaultParameter._parameter.as<std::vector<float>>();
        ASSERT_EQ(expected, actual);
    } else {
        FAIL() << "Unsupported parameter type for key: " << defaultParameter._key;
    }
}
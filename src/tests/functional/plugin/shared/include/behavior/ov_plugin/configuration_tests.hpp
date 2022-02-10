// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include <openvino/runtime/properties.hpp>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"

namespace ov {
namespace test {
namespace behavior {


using CustomComparator = std::function<bool(const Any &, const Any &)>;

struct DefaultParameter {
    std::string _key;
    Any _parameter;
    CustomComparator _comparator;
};

using DefaultConfigurationParameters = std::tuple<
        std::string,    //  device name
        DefaultParameter // default parameter key value comparator
>;

struct DefaultConfigurationTest : public CommonTestUtils::TestsCommon, public ::testing::WithParamInterface<DefaultConfigurationParameters> {
    enum {
        DeviceName, DefaultParamterId
    };

    static std::string getTestCaseName(const ::testing::TestParamInfo<DefaultConfigurationParameters> &obj);

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::string device_name;
    DefaultParameter defaultParameter;
};

class ConfigBase : public CommonTestUtils::TestsCommon {
public:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> model;
    std::string device_name;
    AnyMap configuration;
};

class BehaviorTestsEmptyConfig : public testing::WithParamInterface<std::string>,
                                 public ConfigBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        std::string device_name;
        device_name = obj.param;
        std::ostringstream result;
        result << "device_name=" << device_name;
        return result.str();
    }

    void SetUp() override {        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        device_name = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }
};

using BehaviorParamsSingleOptionDefault = std::tuple<
        std::string,                                       // Device name
        std::pair<std::string, Any> // Configuration key and its default value
>;

class BehaviorTestsSingleOptionDefault : public testing::WithParamInterface<BehaviorParamsSingleOptionDefault>,
                                         public ConfigBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorParamsSingleOptionDefault> obj) {
        std::string device_name;
        std::pair<std::string, Any> configuration;
        std::tie(device_name, configuration) = obj.param;
        std::ostringstream result;
        result << "device_name=" << device_name << "_";
        result << "config=" << "(" << configuration.first << "_" << configuration.second.as<std::string>() << ")";
        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::pair<std::string, Any> entry;
        std::tie(device_name, entry) = this->GetParam();
        std::tie(key, value) = entry;
    }

    std::string key;
    Any value;
};

using CorrectConfigParams = std::tuple<
        std::string,                                      // Device name
        AnyMap // Configuration key and its default value
>;

class CorrectConfigTests : public testing::WithParamInterface<CorrectConfigParams>,
                           public ConfigBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CorrectConfigParams> obj) {
        std::string device_name;
        AnyMap configuration;
        std::tie(device_name, configuration) = obj.param;
        std::ostringstream result;
        result << "device_name=" << device_name << "_";
        if (!configuration.empty()) {
            auto str = util::to_string(configuration);
            std::reaplace(str.begin(), str.end(), ' ', '_');
            result << "config=" << str;
        }
        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(device_name, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
    }
};

using BehaviorParamsSingleOptionCustom = std::tuple<
        std::string,                                                     // Device name
        std::tuple<std::string, std::string, Any> // Configuration key, value and reference
>;

class BehaviorTestsSingleOptionCustom : public testing::WithParamInterface<BehaviorParamsSingleOptionCustom>,
                                        public ConfigBase {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tuple<std::string, std::string, Any> entry;
        std::tie(device_name, entry) = this->GetParam();
        std::tie(key, value, reference) = entry;
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    std::string key;
    std::string value;
    Any reference;
};

using BehaviorParamsSingleOption = std::tuple<
        std::string,                        // Device name
        std::string                         // Key
>;

class BehaviorTestsSingleOption : public testing::WithParamInterface<BehaviorParamsSingleOption>,
                                  public ConfigBase {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(device_name, key) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    std::string key;
};

using OVEmptyConfigTests = BehaviorTestsEmptyConfig;
using OVCorrectSingleOptionDefaultValueConfigTests = BehaviorTestsSingleOptionDefault;
using OVCorrectSingleOptionCustomValueConfigTests = BehaviorTestsSingleOptionCustom;
using OVCorrectConfigPublicOptionsTests = BehaviorTestsSingleOption;
using OVCorrectConfigPrivateOptionsTests = BehaviorTestsSingleOption;
using OVIncorrectConfigTests = CorrectConfigTests;
using OVIncorrectConfigSingleOptionTests = BehaviorTestsSingleOption;
using OVIncorrectConfigAPITests = CorrectConfigTests;
using OVCorrectConfigCheck = CorrectConfigTests;
using OVDefaultValuesConfigTests = CorrectConfigTests;

}  // namespace behavior
}  // namespace test
}  // namespace ov
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <algorithm>

#include <ie_core.hpp>
#include <ie_parameter.hpp>
#include <functional_test_utils/skip_tests_config.hpp>
#include <ov_models/subgraph_builders.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {


using CustomComparator = std::function<bool(const InferenceEngine::Parameter &, const InferenceEngine::Parameter &)>;

struct DefaultParameter {
    std::string _key;
    InferenceEngine::Parameter _parameter;
    CustomComparator _comparator;
};

using DefaultConfigurationParameters = std::tuple<
        std::string,    //  device name
        DefaultParameter // default parameter key value comparator
>;

struct DefaultConfigurationTest : public BehaviorTestsUtils::IEPluginTestBase,
                                  public ::testing::WithParamInterface<DefaultConfigurationParameters> {
    enum {
        DeviceName, DefaultParamterId
    };

    static std::string getTestCaseName(const ::testing::TestParamInfo<DefaultConfigurationParameters> &obj);

protected:
    std::shared_ptr<InferenceEngine::Core> _core = PluginCache::get().ie();
    DefaultParameter defaultParameter;
};

class ConfigBase : public BehaviorTestsUtils::IEPluginTestBase {
public:
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::CNNNetwork cnnNet;
    std::map<std::string, std::string> configuration;
};

class BehaviorTestsEmptyConfig : public testing::WithParamInterface<std::string>,
                                 public ConfigBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        std::string target_device;
        target_device = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');
        std::ostringstream result;
        result << "target_device=" << target_device;
        return result.str();
    }

    void SetUp() override {        // Skip test according to plugin specific disabledTestPatterns() (if any)
        // Create CNNNetwork from ngrpah::Function
        target_device = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }
};

using BehaviorParamsSingleOptionDefault = std::tuple<
        std::string,                                       // Device name
        std::pair<std::string, InferenceEngine::Parameter> // Configuration key and its default value
>;

class BehaviorTestsSingleOptionDefault : public testing::WithParamInterface<BehaviorParamsSingleOptionDefault>,
                                         public ConfigBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorParamsSingleOptionDefault> obj) {
        std::string target_device;
        std::pair<std::string, InferenceEngine::Parameter> configuration;
        std::tie(target_device, configuration) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');
        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        std::string config_value = configuration.second.as<std::string>();
        std::replace(config_value.begin(), config_value.end(), '-', '_');
        result << "config=" << "(" << configuration.first << "_" << config_value << ")";
        return result.str();
    }

    void SetUp() override {
        std::pair<std::string, InferenceEngine::Parameter> entry;
        std::tie(target_device, entry) = this->GetParam();
        std::tie(key, value) = entry;
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
    }

    std::string key;
    InferenceEngine::Parameter value;
};

using CorrectConfigParams = std::tuple<
        std::string,                                      // Device name
        std::map<std::string, std::string> // Configuration key and its default value
>;

class CorrectConfigTests : public testing::WithParamInterface<CorrectConfigParams>,
                           public ConfigBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CorrectConfigParams> obj) {
        std::string target_device;
        std::map<std::string, std::string> configuration;
        std::tie(target_device, configuration) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');
        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            result << "config=" << (configuration);
        }
        return result.str();
    }

    void SetUp() override {
        std::map<std::string, std::string> entry;
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }
};

using BehaviorParamsSingleOptionCustom = std::tuple<
        std::string,                                                     // Device name
        std::tuple<std::string, std::string, InferenceEngine::Parameter> // Configuration key, value and reference
>;

class BehaviorTestsSingleOptionCustom : public testing::WithParamInterface<BehaviorParamsSingleOptionCustom>,
                                        public ConfigBase {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tuple<std::string, std::string, InferenceEngine::Parameter> entry;
        std::tie(target_device, entry) = this->GetParam();
        std::tie(key, value, reference) = entry;
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    std::string key;
    std::string value;
    InferenceEngine::Parameter reference;
};

using BehaviorParamsSingleOption = std::tuple<
        std::string,                        // Device name
        std::string                         // Key
>;

class BehaviorTestsSingleOption : public testing::WithParamInterface<BehaviorParamsSingleOption>,
                                  public ConfigBase {
public:
    void SetUp() override {
        std::tie(target_device, key) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    std::string key;
};

using LoadNetWorkPropertiesParams = std::tuple<
        std::string,                                      // Device name
        std::map<std::string, std::string>,               // Configuration key and its default value
        std::map<std::string, std::string>                // Configuration key and its default value
>;

class SetPropLoadNetWorkGetPropTests : public testing::WithParamInterface<LoadNetWorkPropertiesParams>,
                           public ConfigBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LoadNetWorkPropertiesParams> obj) {
        std::string target_device;
        std::map<std::string, std::string> configuration;
        std::map<std::string, std::string> loadNetWorkConfig;
        std::tie(target_device, configuration, loadNetWorkConfig) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');
        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        if (!configuration.empty()) {
            result << "configItem=";
            for (auto& configItem : configuration) {
                result << configItem.first << "_" << configItem.second << "_";
            }
        }

        if (!loadNetWorkConfig.empty()) {
            result << "loadNetWorkConfig=";
            for (auto& configItem : loadNetWorkConfig) {
                result << configItem.first << "_" << configItem.second << "_";
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::map<std::string, std::string> entry;
        std::tie(target_device, configuration, loadNetWorkConfig) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

public:
    std::map<std::string, std::string> loadNetWorkConfig;
};

using EmptyConfigTests = BehaviorTestsEmptyConfig;
using CorrectSingleOptionDefaultValueConfigTests = BehaviorTestsSingleOptionDefault;
using CorrectSingleOptionCustomValueConfigTests = BehaviorTestsSingleOptionCustom;
using CorrectConfigPublicOptionsTests = BehaviorTestsSingleOption;
using CorrectConfigPrivateOptionsTests = BehaviorTestsSingleOption;
using IncorrectConfigTests = CorrectConfigTests;
using IncorrectConfigSingleOptionTests = BehaviorTestsSingleOption;
using IncorrectConfigAPITests = CorrectConfigTests;
using CorrectConfigCheck = CorrectConfigTests;
using DefaultValuesConfigTests = CorrectConfigTests;

} // namespace BehaviorTestsDefinitions

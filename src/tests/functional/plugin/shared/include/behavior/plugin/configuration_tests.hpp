// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include <ie_core.hpp>
#include <ie_parameter.hpp>
#include <functional_test_utils/skip_tests_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"

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

struct DefaultConfigurationTest : public CommonTestUtils::TestsCommon, public ::testing::WithParamInterface<DefaultConfigurationParameters> {
    enum {
        DeviceName, DefaultParamterId
    };

    static std::string getTestCaseName(const ::testing::TestParamInfo<DefaultConfigurationParameters> &obj);

protected:
    std::shared_ptr<InferenceEngine::Core> _core = PluginCache::get().ie();
    std::string targetDevice;
    DefaultParameter defaultParameter;
};

class ConfigBase : public CommonTestUtils::TestsCommon {
public:
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::CNNNetwork cnnNet;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
};

class BehaviorTestsEmptyConfig : public testing::WithParamInterface<std::string>,
                                 public ConfigBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        std::string targetDevice;
        targetDevice = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void SetUp() override {        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        targetDevice = this->GetParam();
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
        std::string targetDevice;
        std::pair<std::string, InferenceEngine::Parameter> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "config=" << "(" << configuration.first << "_" << configuration.second.as<std::string>() << ")";
        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::pair<std::string, InferenceEngine::Parameter> entry;
        std::tie(targetDevice, entry) = this->GetParam();
        std::tie(key, value) = entry;
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
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            using namespace CommonTestUtils;
            result << "config=" << (configuration);
        }
        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::map<std::string, std::string> entry;
        std::tie(targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
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
        std::tie(targetDevice, entry) = this->GetParam();
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
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(targetDevice, key) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    std::string key;
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
using ExclusiveAsyncReqTests = CorrectConfigTests;

} // namespace BehaviorTestsDefinitions
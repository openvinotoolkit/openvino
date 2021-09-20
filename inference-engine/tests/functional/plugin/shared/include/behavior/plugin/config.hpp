// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/behavior_test_utils.hpp"

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "ie_extension.h"
#include <condition_variable>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_plugin_config.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <gna/gna_config.hpp>
#include <ie_core.hpp>
#include "ie_common.h"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include <threading/ie_executor_manager.hpp>
#include <base/behavior_test_utils.hpp>
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace BehaviorTestsDefinitions {

using namespace CommonTestUtils;

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
        std::tie(targetDevice) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void SetUp()  override {        // Skip test according to plugin specific disabledTestPatterns() (if any)
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
                                         public ConfigBase  {
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

    void SetUp()  override {
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
            result << "config=" << configuration;
        }
        return result.str();
    }

    void SetUp()  override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::map<std::string, std::string> entry;
        std::tie(targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }
};

using BehaviorParamsSingleOptionCustom = std::tuple<
        InferenceEngine::Precision,                                      // Network precision
        std::string,                                                     // Device name
        std::tuple<std::string, std::string, InferenceEngine::Parameter> // Configuration key, value and reference
>;

class BehaviorTestsSingleOptionCustom : public testing::WithParamInterface<BehaviorParamsSingleOptionCustom>,
                                        public ConfigBase {
public:
    void SetUp()  override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tuple<std::string, std::string, InferenceEngine::Parameter> entry;
        std::tie(netPrecision, targetDevice, entry) = this->GetParam();
        std::tie(key, value, reference) = entry;
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    InferenceEngine::Precision netPrecision;
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
    void SetUp()  override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(targetDevice, key) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    std::string key;
};

using EmptyConfigTests = BehaviorTestsEmptyConfig;
// Setting empty config doesn't throw
TEST_P(EmptyConfigTests, SetEmptyConfig) {
    std::map<std::string, std::string> config;
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
}

TEST_P(EmptyConfigTests, CanLoadNetworkWithEmptyConfig) {
    std::map<std::string, std::string> config;
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, config));
}

using CorrectSingleOptionDefaultValueConfigTests = BehaviorTestsSingleOptionDefault;
TEST_P(CorrectSingleOptionDefaultValueConfigTests, CheckDefaultValueOfConfig) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_EQ(ie->GetConfig(targetDevice, key), value);
}

// Setting correct config doesn't throw
TEST_P(CorrectConfigTests, SetCorrectConfig) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
}

TEST_P(CorrectConfigTests, CanLoadNetworkWithCorrectConfig) {
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
}

TEST_P(CorrectConfigTests, CanUseCache) {
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    ie->SetConfig({ { CONFIG_KEY(CACHE_DIR), "./test_cache" } });
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
    CommonTestUtils::removeDir("./test_cache");
}

using CorrectSingleOptionCustomValueConfigTests = BehaviorTestsSingleOptionCustom;
TEST_P(CorrectSingleOptionCustomValueConfigTests, CheckCustomValueOfConfig) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::map<std::string, std::string> configuration = {{key, value}};
    ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
    ASSERT_EQ(ie->GetConfig(targetDevice, key), reference);
}

using CorrectConfigPublicOptionsTests = BehaviorTestsSingleOption;
TEST_P(CorrectConfigPublicOptionsTests, CanSeePublicOption) {
    InferenceEngine::Parameter metric;
    ASSERT_NO_THROW(metric = ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    const auto& supportedOptions = metric.as<std::vector<std::string>>();
    ASSERT_NE(std::find(supportedOptions.cbegin(), supportedOptions.cend(), key), supportedOptions.cend());
}

using CorrectConfigPrivateOptionsTests = BehaviorTestsSingleOption;
TEST_P(CorrectConfigPrivateOptionsTests, CanNotSeePrivateOption) {
    InferenceEngine::Parameter metric;
    ASSERT_NO_THROW(metric = ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    const auto& supportedOptions = metric.as<std::vector<std::string>>();
    ASSERT_EQ(std::find(supportedOptions.cbegin(), supportedOptions.cend(), key), supportedOptions.cend());
}

using IncorrectConfigTests = CorrectConfigTests;
TEST_P(IncorrectConfigTests, SetConfigWithIncorrectKey) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_THROW(ie->SetConfig(configuration, targetDevice), InferenceEngine::Exception);
}

TEST_P(IncorrectConfigTests, CanNotLoadNetworkWithIncorrectConfig) {
    ASSERT_THROW(auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration),
                    InferenceEngine::Exception);
}

using IncorrectConfigSingleOptionTests = BehaviorTestsSingleOption;
TEST_P(IncorrectConfigSingleOptionTests, CanNotGetConfigWithIncorrectConfig) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_THROW(ie->GetConfig(targetDevice, key), InferenceEngine::Exception);
}

using IncorrectConfigAPITests = CorrectConfigTests;
TEST_P(IncorrectConfigAPITests, SetConfigWithNoExistingKey) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    if (targetDevice.find(CommonTestUtils::DEVICE_GNA) != std::string::npos) {
        ASSERT_THROW(ie->SetConfig(configuration, targetDevice), InferenceEngine::NotFound);
    } else {
        ASSERT_THROW(ie->SetConfig(configuration, targetDevice), InferenceEngine::Exception);
    }
}
}  // namespace BehaviorTestsDefinitions

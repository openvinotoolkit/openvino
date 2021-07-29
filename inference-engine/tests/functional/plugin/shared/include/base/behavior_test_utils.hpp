// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeindex>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/node.hpp>
#include <ngraph/function.hpp>
#include <ie_plugin_config.hpp>
#include <ngraph/function.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include "gtest/gtest.h"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

namespace BehaviorTestsUtils {

using BehaviorParamsEmptyConfig = std::tuple<
    InferenceEngine::Precision,         // Network precision
    std::string                         // Device name
>;

class BehaviorTestsEmptyConfig : public testing::WithParamInterface<BehaviorParamsEmptyConfig>,
                                 public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorParamsEmptyConfig> obj) {
        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::tie(netPrecision, targetDevice) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void SetUp()  override {
        std::tie(netPrecision, targetDevice) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void TearDown() override {
        function.reset();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
};

typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> BehaviorParams;

class BehaviorTestsBasic : public testing::WithParamInterface<BehaviorParams>,
                           public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorParams> obj) {
        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    void SetUp()  override {
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        function.reset();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
};

using BehaviorParamsSingleOption = std::tuple<
    InferenceEngine::Precision,         // Network precision
    std::string,                        // Device name
    std::string                         // Key
>;

class BehaviorTestsSingleOption : public testing::WithParamInterface<BehaviorParamsSingleOption>,
                                  public CommonTestUtils::TestsCommon {
public:
    void SetUp()  override {
        std::tie(netPrecision, targetDevice, key) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void TearDown() override {
        function.reset();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::string key;
};

using BehaviorParamsSingleOptionDefault = std::tuple<
    InferenceEngine::Precision,                        // Network precision
    std::string,                                       // Device name
    std::pair<std::string, InferenceEngine::Parameter> // Configuration key and its default value
>;

class BehaviorTestsSingleOptionDefault : public testing::WithParamInterface<BehaviorParamsSingleOptionDefault>,
                                         public CommonTestUtils::TestsCommon {
public:
    void SetUp()  override {
        std::pair<std::string, InferenceEngine::Parameter> entry;
        std::tie(netPrecision, targetDevice, entry) = this->GetParam();
        std::tie(key, value) = entry;
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void TearDown() override {
        function.reset();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::string key;
    InferenceEngine::Parameter value;
};

using BehaviorParamsSingleOptionCustom = std::tuple<
    InferenceEngine::Precision,                                      // Network precision
    std::string,                                                     // Device name
    std::tuple<std::string, std::string, InferenceEngine::Parameter> // Configuration key, value and reference
>;

class BehaviorTestsSingleOptionCustom : public testing::WithParamInterface<BehaviorParamsSingleOptionCustom>,
                                        public CommonTestUtils::TestsCommon {
public:
    void SetUp()  override {
        std::tuple<std::string, std::string, InferenceEngine::Parameter> entry;
        std::tie(netPrecision, targetDevice, entry) = this->GetParam();
        std::tie(key, value, reference) = entry;
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void TearDown() override {
        function.reset();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::string key;
    std::string value;
    InferenceEngine::Parameter reference;
};

}  // namespace BehaviorTestsUtils

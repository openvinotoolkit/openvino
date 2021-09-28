// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "openvino/runtime/runtime.hpp"

namespace ov {
namespace test {

using BehaviorParamsEmptyConfig = std::tuple<ov::element::Type,  // element type
                                             std::string>;       // device

class BehaviorTestsEmptyConfig : public testing::WithParamInterface<BehaviorParamsEmptyConfig>,
                                 public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorParamsEmptyConfig> obj) {
        std::string targetDevice;
        ov::element::Type elementType;
        std::tie(elementType, targetDevice) = obj.param;
        std::ostringstream result;
        result << "element_type=" << elementType;
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void SetUp() override {
        std::tie(elementType, targetDevice) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu({1, 1, 32, 32}, elementType);
    }

    void TearDown() override {
        function.reset();
    }

    std::shared_ptr<ov::runtime::Core> ie = PluginCache::get().core();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    ov::element::Type elementType;
};

using BehaviorBasicParams = std::tuple<ov::element::Type,                    // element type
                                       std::string,                          // device
                                       std::map<std::string, std::string>>;  // config

class BehaviorTestsBasic : public testing::WithParamInterface<BehaviorBasicParams>,
                           public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorBasicParams> obj) {
        std::string targetDevice;
        ov::element::Type elementType;
        std::map<std::string, std::string> configuration;
        std::tie(elementType, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "element_type=" << elementType;
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(elementType, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu({1, 1, 32, 32}, elementType);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        function.reset();
    }

    std::shared_ptr<ov::runtime::Core> ie = PluginCache::get().core();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    ov::element::Type elementType;
};

using InferRequestParams = std::tuple<ov::element::Type,                    // element type
                                      std::string,                          // device
                                      std::map<std::string, std::string>>;  // config

class InferRequestTests : public testing::WithParamInterface<InferRequestParams>, public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        std::string targetDevice;
        ov::element::Type elementType;
        std::map<std::string, std::string> configuration;
        std::tie(elementType, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "element_type=" << elementType;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    void SetUp() override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(elementType, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu({1, 1, 32, 32}, elementType);
        // Load CNNNetwork to target plugins
        execNet = ie->compile_model(function, targetDevice, configuration);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        function.reset();
    }

protected:
    ov::runtime::ExecutableNetwork execNet;
    std::shared_ptr<ov::runtime::Core> ie = PluginCache::get().core();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    ov::element::Type elementType;
};

using BehaviorParamsSingleOption = std::tuple<ov::element::Type,  // element type
                                              std::string,        // device
                                              std::string>;       // key

class BehaviorTestsSingleOption : public testing::WithParamInterface<BehaviorParamsSingleOption>,
                                  public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        std::tie(elementType, targetDevice, key) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu({1, 1, 32, 32}, elementType);
    }

    void TearDown() override {
        function.reset();
    }

    std::shared_ptr<ov::runtime::Core> ie = PluginCache::get().core();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    std::string key;
    ov::element::Type elementType;
};

using BehaviorParamsSingleOptionDefault =
    std::tuple<ov::element::Type,                                  // element type
               std::string,                                        // Device name
               std::pair<std::string, InferenceEngine::Parameter>  // Configuration key and its default value
               >;

class BehaviorTestsSingleOptionDefault : public testing::WithParamInterface<BehaviorParamsSingleOptionDefault>,
                                         public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        std::pair<std::string, InferenceEngine::Parameter> entry;
        std::tie(elementType, targetDevice, entry) = this->GetParam();
        std::tie(key, value) = entry;
        function = ngraph::builder::subgraph::makeConvPoolRelu({1, 1, 32, 32}, elementType);
    }

    void TearDown() override {
        function.reset();
    }

    std::shared_ptr<ov::runtime::Core> ie = PluginCache::get().core();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    std::string key;
    InferenceEngine::Parameter value;
    ov::element::Type elementType;
};

using BehaviorParamsSingleOptionCustom =
    std::tuple<ov::element::Type,                                                // element type
               std::string,                                                      // Device name
               std::tuple<std::string, std::string, InferenceEngine::Parameter>  // Configuration key, value and
                                                                                 // reference
               >;

class BehaviorTestsSingleOptionCustom : public testing::WithParamInterface<BehaviorParamsSingleOptionCustom>,
                                        public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        std::tuple<std::string, std::string, InferenceEngine::Parameter> entry;
        std::tie(elementType, targetDevice, entry) = this->GetParam();
        std::tie(key, value, reference) = entry;
        function = ngraph::builder::subgraph::makeConvPoolRelu({1, 1, 32, 32}, elementType);
    }

    void TearDown() override {
        function.reset();
    }

    std::shared_ptr<ov::runtime::Core> ie = PluginCache::get().core();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    std::string key;
    std::string value;
    ov::runtime::Parameter reference;
    ov::element::Type elementType;
};

}  // namespace test
}  // namespace ov

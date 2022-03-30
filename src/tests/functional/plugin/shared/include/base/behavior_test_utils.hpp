// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_behavior_test_utils.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/summary/api_summary.hpp"

namespace BehaviorTestsUtils {

using namespace CommonTestUtils;

typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> BehaviorBasicParams;

class BehaviorTestsBasic : public testing::WithParamInterface<BehaviorBasicParams>,
                           public ov::test::behavior::APIBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorBasicParams> obj) {
        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            result << "config=" << configuration;
        }
        return result.str();
    }

    void SetUp() override {
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(netPrecision, target_device, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }
    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    ov::test::utils::ov_entity api_entity;
};

typedef std::tuple<
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> InferRequestParams;

class InferRequestTests : public testing::WithParamInterface<InferRequestParams>,
                          public ov::test::behavior::APIBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            for (auto &configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    void SetUp() override {
        APIBaseTest::SetUp();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(target_device, configuration) = this->GetParam();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(target_device);
        cnnNet = InferenceEngine::CNNNetwork(function);
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    }
    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

protected:
    void set_api_entity() override { api_entity = ov::test::utils::ov_entity::ie_infer_request; };

    InferenceEngine::CNNNetwork cnnNet;
    InferenceEngine::ExecutableNetwork execNet;
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    std::map<std::string, std::string> configuration;
    ov::test::utils::ov_entity api_entity = ov::test::utils::ov_entity::ie_infer_request;
};

inline InferenceEngine::Core createIECoreWithTemplate() {
    PluginCache::get().reset();
    InferenceEngine::Core ie;
#ifndef OPENVINO_STATIC_LIBRARY
    std::string pluginName = "openvino_template_plugin";
    pluginName += IE_BUILD_POSTFIX;
    ie.RegisterPlugin(pluginName, CommonTestUtils::DEVICE_TEMPLATE);
#endif // !OPENVINO_STATIC_LIBRARY
    return ie;
}

class IEClassNetworkTest : public ov::test::behavior::OVClassNetworkTest {
public:
    InferenceEngine::CNNNetwork actualCnnNetwork, simpleCnnNetwork, multinputCnnNetwork, ksoCnnNetwork;

    void SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVClassNetworkTest::SetUp();
        // Generic network
        ASSERT_NO_THROW(actualCnnNetwork = InferenceEngine::CNNNetwork(actualNetwork));
        // Quite simple network
        ASSERT_NO_THROW(simpleCnnNetwork = InferenceEngine::CNNNetwork(simpleNetwork));
        // Multinput to substruct network
        ASSERT_NO_THROW(multinputCnnNetwork = InferenceEngine::CNNNetwork(multinputNetwork));
        // Network with KSO
        ASSERT_NO_THROW(ksoCnnNetwork = InferenceEngine::CNNNetwork(ksoNetwork));
    }
};

class IEClassBaseTestP : public IEClassNetworkTest,
                         public ::testing::WithParamInterface<std::string>,
                         public ov::test::behavior::APIBaseTest {
public:
    void SetUp() override {
        APIBaseTest::SetUp();
        IEClassNetworkTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        target_device = GetParam();
    }

protected:
    void set_api_entity() override { api_entity = ov::test::utils::ov_entity::ie_plugin; };
};
} // namespace BehaviorTestsUtils

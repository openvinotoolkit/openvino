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
                           public CommonTestUtils::TestsCommon {
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

    virtual void setApiEntity() { api_entity = ov::test::utils::ov_entity::UNDEFINED; }

    void SetUp() override {
        setApiEntity();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        auto& apiSummary = ov::test::utils::ApiSummary::getInstance();
        if (this->HasFailure()) {
            apiSummary.updateStat(api_entity, targetDevice, ov::test::utils::PassRate::Statuses::FAILED);
        } else if (this->IsSkipped()) {
            apiSummary.updateStat(api_entity, targetDevice, ov::test::utils::PassRate::Statuses::SKIPPED);
        } else {
            apiSummary.updateStat(api_entity, targetDevice, ov::test::utils::PassRate::Statuses::PASSED);
        }
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
                          public CommonTestUtils::TestsCommon {
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
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(targetDevice, configuration) = this->GetParam();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(targetDevice);
        cnnNet = InferenceEngine::CNNNetwork(function);
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        auto& apiSummary = ov::test::utils::ApiSummary::getInstance();
        if (this->HasFailure()) {
            apiSummary.updateStat(api_entity, targetDevice, ov::test::utils::PassRate::Statuses::FAILED);
        } else if (this->IsSkipped()) {
            apiSummary.updateStat(api_entity, targetDevice, ov::test::utils::PassRate::Statuses::SKIPPED);
        } else {
            apiSummary.updateStat(api_entity, targetDevice, ov::test::utils::PassRate::Statuses::PASSED);
        }
    }

protected:
    InferenceEngine::CNNNetwork cnnNet;
    InferenceEngine::ExecutableNetwork execNet;
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
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

    void SetUp() override {
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

class IEClassBaseTestP : public IEClassNetworkTest, public ::testing::WithParamInterface<std::string> {
public:
    std::string deviceName;
    ov::test::utils::ov_entity api_entity = ov::test::utils::ov_entity::UNDEFINED;

    void SetUp() override {
        api_entity = ov::test::utils::ov_entity::ie_plugin;
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        IEClassNetworkTest::SetUp();
        deviceName = GetParam();
    }

    void TearDown() override {
        auto &apiSummary = ov::test::utils::ApiSummary::getInstance();
        if (this->HasFailure()) {
            apiSummary.updateStat(api_entity, deviceName, ov::test::utils::PassRate::Statuses::FAILED);
        } else if (this->IsSkipped()) {
            apiSummary.updateStat(api_entity, deviceName, ov::test::utils::PassRate::Statuses::SKIPPED);
        } else {
            apiSummary.updateStat(api_entity, deviceName, ov::test::utils::PassRate::Statuses::PASSED);
        }
    }
};
} // namespace BehaviorTestsUtils

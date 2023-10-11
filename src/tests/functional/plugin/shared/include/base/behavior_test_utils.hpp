// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_behavior_test_utils.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"
#include "functional_test_utils/summary/api_summary.hpp"

namespace BehaviorTestsUtils {

class IEInferRequestTestBase :  public ov::test::behavior::APIBaseTest {
private:
    void set_api_entity() override {
        api_entity = ov::test::utils::ov_entity::ie_infer_request;
    };
};

class IEExecutableNetworkTestBase :  public ov::test::behavior::APIBaseTest {
private:
    void set_api_entity() override {
        api_entity = ov::test::utils::ov_entity::ie_executable_network;
    };
};

class IEPluginTestBase :  public ov::test::behavior::APIBaseTest {
private:
    void set_api_entity() override {
        api_entity = ov::test::utils::ov_entity::ie_plugin;
    };
};

typedef std::tuple<
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> InferRequestParams;

class InferRequestTests : public testing::WithParamInterface<InferRequestParams>,
                          public IEInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::ostringstream result;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            for (auto &configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
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
    InferenceEngine::CNNNetwork cnnNet;
    InferenceEngine::ExecutableNetwork execNet;
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    std::map<std::string, std::string> configuration;;
};

inline InferenceEngine::Core createIECoreWithTemplate() {
    PluginCache::get().reset();
    InferenceEngine::Core ie;
#ifndef OPENVINO_STATIC_LIBRARY
    std::string pluginName = "openvino_template_plugin" OV_BUILD_POSTFIX;
    ie.RegisterPlugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(), pluginName),
        ov::test::utils::DEVICE_TEMPLATE);
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
                         public IEPluginTestBase {
public:
    void SetUp() override {
        target_device = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        IEClassNetworkTest::SetUp();
    }
};

class IEExecNetClassBaseTestP : public IEClassNetworkTest,
                                public ::testing::WithParamInterface<std::string>,
                                public IEExecutableNetworkTestBase {
public:
    void SetUp() override {
        target_device = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        IEClassNetworkTest::SetUp();
    }
};

typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> BehaviorBasicParams;

class BehaviorTestsBasicBase : public testing::WithParamInterface<BehaviorBasicParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorBasicParams> obj) {
        using namespace ov::test::utils;

        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            result << "config=" << configuration;
        }
        return result.str();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> configuration;
};

class BehaviorTestsBasic : public BehaviorTestsBasicBase,
                           public IEPluginTestBase {
protected:
    void SetUp() override {
        std::tie(netPrecision, target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }
    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }
};
} // namespace BehaviorTestsUtils

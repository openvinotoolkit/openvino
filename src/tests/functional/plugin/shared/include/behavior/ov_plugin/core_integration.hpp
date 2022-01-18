// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>
#endif

namespace ov {
namespace test {
namespace behavior {

#define ASSERT_METRIC_SUPPORTED(metricName)                                                      \
{                                                                                                \
    std::vector<std::string> metrics = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_METRICS)); \
    auto it = std::find(metrics.begin(), metrics.end(), metricName);                             \
    ASSERT_NE(metrics.end(), it);                                                                \
}


class OVClassBasicTestP : public ::testing::Test, public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string pluginName;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(pluginName, deviceName) = GetParam();
        pluginName += IE_BUILD_POSTFIX;
    }
};

class OVClassSetDefaultDeviceIDTest : public ::testing::Test,
                                      public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string deviceID;
public:
    void SetUp() override {
        std::tie(deviceName, deviceID) = GetParam();
    }
};

using OVClassNetworkTestP = OVClassBaseTestP;
using OVClassQueryNetworkTest = OVClassBaseTestP;
using OVClassImportExportTestP = OVClassBaseTestP;
using OVClassGetMetricTest_SUPPORTED_METRICS = OVClassBaseTestP;
using OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassBaseTestP;
using OVClassGetMetricTest_AVAILABLE_DEVICES = OVClassBaseTestP;
using OVClassGetMetricTest_FULL_DEVICE_NAME = OVClassBaseTestP;
using OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES = OVClassBaseTestP;
using OVClassGetMetricTest_DEVICE_GOPS = OVClassBaseTestP;
using OVClassGetMetricTest_DEVICE_TYPE = OVClassBaseTestP;
using OVClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS = OVClassBaseTestP;
using OVClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS = OVClassBaseTestP;
using OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS = OVClassBaseTestP;
using OVClassGetMetricTest_ThrowUnsupported = OVClassBaseTestP;
using OVClassGetConfigTest = OVClassBaseTestP;
using OVClassGetConfigTest_ThrowUnsupported = OVClassBaseTestP;
using OVClassGetAvailableDevices = OVClassBaseTestP;
using OVClassGetMetricTest_RANGE_FOR_STREAMS = OVClassBaseTestP;
using OVClassLoadNetworkAfterCoreRecreateTest = OVClassBaseTestP;
using OVClassLoadNetworkTest = OVClassQueryNetworkTest;
using OVClassSetGlobalConfigTest = OVClassBaseTestP;
using OVClassSpecificDeviceTestSetConfig = OVClassBaseTestP;
using OVClassSpecificDeviceTestGetConfig = OVClassBaseTestP;

class OVClassSeveralDevicesTest : public OVClassNetworkTest,
                                  public ::testing::WithParamInterface<std::vector<std::string>> {
public:
    std::vector<std::string> deviceNames;
    void SetUp() override {
        OVClassNetworkTest::SetUp();
        deviceNames = GetParam();
    }
};
using OVClassSeveralDevicesTestLoadNetwork = OVClassSeveralDevicesTest;
using OVClassSeveralDevicesTestQueryNetwork = OVClassSeveralDevicesTest;
using OVClassSeveralDevicesTestDefaultCore = OVClassSeveralDevicesTest;

inline bool supportsAvaliableDevices(ov::runtime::Core& ie, const std::string& deviceName) {
    auto supportedMetricKeys = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
    return supportedMetricKeys.end() !=
           std::find(std::begin(supportedMetricKeys), std::end(supportedMetricKeys), METRIC_KEY(AVAILABLE_DEVICES));
}

bool supportsDeviceID(ov::runtime::Core& ie, const std::string& deviceName) {
    auto supportedConfigKeys =
            ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
    return supportedConfigKeys.end() !=
           std::find(std::begin(supportedConfigKeys), std::end(supportedConfigKeys), CONFIG_KEY(DEVICE_ID));
}

TEST(OVClassBasicTest, smoke_createDefault) {
    ASSERT_NO_THROW(ov::runtime::Core ie);
}

TEST_P(OVClassBasicTestP, registerExistingPluginThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.register_plugin(pluginName, deviceName), ov::Exception);
}

// TODO: CVS-68982
#ifndef OPENVINO_STATIC_LIBRARY

TEST_P(OVClassBasicTestP, registerNewPluginNoThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.register_plugin(pluginName, "NEW_DEVICE_NAME"));
    ASSERT_NO_THROW(ie.get_metric("NEW_DEVICE_NAME", METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
}

TEST(OVClassBasicTest, smoke_registerExistingPluginFileThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.register_plugins("nonExistPlugins.xml"), ov::Exception);
    ASSERT_THROW(ie.register_plugins("nonExistPlugins.xml"), ov::Exception);
}

TEST(OVClassBasicTest, smoke_createNonExistingConfigThrows) {
    ASSERT_THROW(ov::runtime::Core ie("nonExistPlugins.xml"), ov::Exception);
}

#ifdef __linux__

TEST(OVClassBasicTest, smoke_createMockEngineConfigNoThrows) {
    std::string filename{"mock_engine_valid.xml"};
    std::string content{"<ie><plugins><plugin name=\"mock\" location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_NO_THROW(ov::runtime::Core ie(filename));
    CommonTestUtils::removeFile(filename.c_str());
}

TEST(OVClassBasicTest, smoke_createMockEngineConfigThrows) {
    std::string filename{"mock_engine.xml"};
    std::string content{"<ie><plugins><plugin location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_THROW(ov::runtime::Core ie(filename), ov::Exception);
    CommonTestUtils::removeFile(filename.c_str());
}

#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

TEST_P(OVClassBasicTestP, smoke_registerPluginsXMLUnicodePath) {
    std::string pluginXML{"mock_engine_valid.xml"};
    std::string content{"<ie><plugins><plugin name=\"mock\" location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(pluginXML, content);

    for (std::size_t testIndex = 0; testIndex < CommonTestUtils::test_unicode_postfix_vector.size(); testIndex++) {
        GTEST_COUT << testIndex;
        std::wstring postfix = L"_" + CommonTestUtils::test_unicode_postfix_vector[testIndex];
        std::wstring pluginsXmlW = CommonTestUtils::addUnicodePostfixToPath(pluginXML, postfix);

        try {
            bool is_copy_successfully;
            is_copy_successfully = CommonTestUtils::copyFile(pluginXML, pluginsXmlW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << pluginXML << "' to '"
                       << ::ov::util::wstring_to_string(pluginsXmlW) << "'";
            }

            GTEST_COUT << "Test " << testIndex << std::endl;

            ov::runtime::Core ie = createCoreWithTemplate();
            GTEST_COUT << "Core created " << testIndex << std::endl;
            ASSERT_NO_THROW(ie.register_plugins(::ov::util::wstring_to_string(pluginsXmlW)));
            CommonTestUtils::removeFile(pluginsXmlW);
#    if defined __linux__ && !defined(__APPLE__)
            ASSERT_NO_THROW(ie.get_versions("mock"));  // from pluginXML
#    endif
            ASSERT_NO_THROW(ie.get_versions(deviceName));
            GTEST_COUT << "Plugin created " << testIndex << std::endl;

            ASSERT_NO_THROW(ie.register_plugin(pluginName, "TEST_DEVICE"));
            ASSERT_NO_THROW(ie.get_versions("TEST_DEVICE"));
            GTEST_COUT << "Plugin registered and created " << testIndex << std::endl;

            GTEST_COUT << "OK" << std::endl;
        } catch (const ov::Exception& e_next) {
            CommonTestUtils::removeFile(pluginsXmlW);
            std::remove(pluginXML.c_str());
            FAIL() << e_next.what();
        }
    }
    CommonTestUtils::removeFile(pluginXML);
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#endif // !OPENVINO_STATIC_LIBRARY

//
// GetVersions()
//

TEST_P(OVClassBasicTestP, getVersionsByExactDeviceNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.get_versions(deviceName + ".0"));
}

TEST_P(OVClassBasicTestP, getVersionsByDeviceClassNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.get_versions(deviceName));
}

TEST_P(OVClassBasicTestP, getVersionsNonEmpty) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_EQ(2, ie.get_versions(CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName).size());
}

//
// UnregisterPlugin
//

TEST_P(OVClassBasicTestP, unregisterExistingPluginNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    // device instance is not created yet
    ASSERT_THROW(ie.unload_plugin(deviceName), ov::Exception);

    // make the first call to IE which created device instance
    ie.get_versions(deviceName);
    // now, we can unregister device
    ASSERT_NO_THROW(ie.unload_plugin(deviceName));
}

TEST_P(OVClassBasicTestP, accessToUnregisteredPluginThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.unload_plugin(deviceName), ov::Exception);
    ASSERT_NO_THROW(ie.get_versions(deviceName));
    ASSERT_NO_THROW(ie.unload_plugin(deviceName));
    ASSERT_NO_THROW(ie.set_config({}, deviceName));
    ASSERT_NO_THROW(ie.get_versions(deviceName));
    ASSERT_NO_THROW(ie.unload_plugin(deviceName));
}

TEST(OVClassBasicTest, smoke_unregisterNonExistingPluginThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.unload_plugin("unkown_device"), ov::Exception);
}

//
// SetConfig
//

TEST_P(OVClassBasicTestP, SetConfigAllThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{"unsupported_key", "4"}}));
    ASSERT_ANY_THROW(ie.get_versions(deviceName));
}

TEST_P(OVClassBasicTestP, SetConfigForUnRegisteredDeviceThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.set_config({{"unsupported_key", "4"}}, "unregistered_device"), ov::Exception);
}

TEST_P(OVClassBasicTestP, SetConfigNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}},
                                  deviceName));
}

TEST_P(OVClassBasicTestP, SetConfigAllNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}}));
    ASSERT_NO_THROW(ie.get_versions(deviceName));
}

TEST(OVClassBasicTest, smoke_SetConfigHeteroThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}},
                                  CommonTestUtils::DEVICE_HETERO));
}

TEST_P(OVClassBasicTestP, SetConfigHeteroTargetFallbackThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{"TARGET_FALLBACK", deviceName}}, CommonTestUtils::DEVICE_HETERO));
}

TEST(OVClassBasicTest, smoke_SetConfigHeteroNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    bool value = false;

    ASSERT_NO_THROW(ie.set_config({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), InferenceEngine::PluginConfigParams::YES}},
                                  CommonTestUtils::DEVICE_HETERO));
    ASSERT_NO_THROW(value = ie.get_config("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_TRUE(value);

    ASSERT_NO_THROW(ie.set_config({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), InferenceEngine::PluginConfigParams::NO}},
                                  CommonTestUtils::DEVICE_HETERO));
    ASSERT_NO_THROW(value = ie.get_config("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_FALSE(value);
}

TEST_P(OVClassSpecificDeviceTestSetConfig, SetConfigSpecificDeviceNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    std::string deviceID, clearDeviceName;
    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceName.substr(0, pos);
        deviceID =  deviceName.substr(pos + 1,  deviceName.size());
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    std::vector<std::string> deviceIDs = ie.get_metric(clearDeviceName, METRIC_KEY(AVAILABLE_DEVICES));
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_SKIP();
    }

    ASSERT_NO_THROW(ie.set_config({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}}, deviceName));
    std::string value;
    ASSERT_NO_THROW(value = ie.get_config(deviceName, InferenceEngine::PluginConfigParams::KEY_PERF_COUNT).as<std::string>());
    ASSERT_EQ(value, InferenceEngine::PluginConfigParams::YES);
}

//
// QueryNetwork
//

TEST_P(OVClassNetworkTestP, QueryNetworkActualThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.query_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(OVClassNetworkTestP, QueryNetworkActualNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    try {
        ie.query_model(actualNetwork, deviceName);
    } catch (const ov::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassNetworkTestP, QueryNetworkWithKSO) {
    ov::runtime::Core ie = createCoreWithTemplate();

    try {
        auto rl_map = ie.query_model(ksoNetwork, deviceName);
        auto func = ksoNetwork;
        for (const auto& op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
    } catch (const ov::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassSeveralDevicesTestQueryNetwork, QueryNetworkActualSeveralDevicesNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    std::string clearDeviceName;
    auto pos = deviceNames.begin()->find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceNames.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    std::vector<std::string> deviceIDs = ie.get_metric(clearDeviceName, METRIC_KEY(AVAILABLE_DEVICES));
    if (deviceIDs.size() < deviceNames.size())
        GTEST_SKIP();

    std::string multiDeviceName = CommonTestUtils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : deviceNames) {
        multiDeviceName += dev_name;
        if (&dev_name != &(deviceNames.back())) {
            multiDeviceName += ",";
        }
    }
    ASSERT_NO_THROW(ie.query_model(actualNetwork, multiDeviceName));
}

TEST_P(OVClassNetworkTestP, SetAffinityWithConstantBranches) {
    ov::runtime::Core ie = createCoreWithTemplate();

    try {
        std::shared_ptr<ngraph::Function> func;
        {
            ngraph::PartialShape shape({1, 84});
            ngraph::element::Type type(ngraph::element::Type_t::f32);
            auto param = std::make_shared<ngraph::opset6::Parameter>(type, shape);
            auto matMulWeights = ngraph::opset6::Constant::create(ngraph::element::Type_t::f32, {10, 84}, {1});
            auto shapeOf = std::make_shared<ngraph::opset6::ShapeOf>(matMulWeights);
            auto gConst1 = ngraph::opset6::Constant::create(ngraph::element::Type_t::i32, {1}, {1});
            auto gConst2 = ngraph::opset6::Constant::create(ngraph::element::Type_t::i64, {}, {0});
            auto gather = std::make_shared<ngraph::opset6::Gather>(shapeOf, gConst1, gConst2);
            auto concatConst = ngraph::opset6::Constant::create(ngraph::element::Type_t::i64, {1}, {1});
            auto concat = std::make_shared<ngraph::opset6::Concat>(ngraph::NodeVector{concatConst, gather}, 0);
            auto relu = std::make_shared<ngraph::opset6::Relu>(param);
            auto reshape = std::make_shared<ngraph::opset6::Reshape>(relu, concat, false);
            auto matMul = std::make_shared<ngraph::opset6::MatMul>(reshape, matMulWeights, false, true);
            auto matMulBias = ngraph::opset6::Constant::create(ngraph::element::Type_t::f32, {1, 10}, {1});
            auto addBias = std::make_shared<ngraph::opset6::Add>(matMul, matMulBias);
            auto result = std::make_shared<ngraph::opset6::Result>(addBias);

            ngraph::ParameterVector params = {param};
            ngraph::ResultVector results = {result};

            func = std::make_shared<ngraph::Function>(results, params);
        }

        auto rl_map = ie.query_model(func, deviceName);
        for (const auto& op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
        for (const auto& op : func->get_ops()) {
            std::string affinity = rl_map[op->get_friendly_name()];
            op->get_rt_info()["affinity"] = affinity;
        }
        auto exeNetwork = ie.compile_model(ksoNetwork, deviceName);
    } catch (const InferenceEngine::NotImplemented& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassNetworkTestP, SetAffinityWithKSO) {
    ov::runtime::Core ie = createCoreWithTemplate();

    try {
        auto rl_map = ie.query_model(ksoNetwork, deviceName);
        auto func = ksoNetwork;
        for (const auto& op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
        for (const auto& op : func->get_ops()) {
            std::string affinity = rl_map[op->get_friendly_name()];
            op->get_rt_info()["affinity"] = affinity;
        }
        auto exeNetwork = ie.compile_model(ksoNetwork, deviceName);
    } catch (const ov::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassNetworkTestP, QueryNetworkHeteroActualNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::runtime::SupportedOpsMap res;
    ASSERT_NO_THROW(
            res = ie.query_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK", deviceName}}));
    ASSERT_LT(0, res.size());
}

TEST_P(OVClassNetworkTestP, QueryNetworkMultiThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.query_model(actualNetwork, CommonTestUtils::DEVICE_MULTI), ov::Exception);
}

TEST(OVClassBasicTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;
    std::string deviceName = CommonTestUtils::DEVICE_HETERO;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> t = p;

    std::cout << "Supported HETERO metrics: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST(OVClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;
    std::string deviceName = CommonTestUtils::DEVICE_HETERO;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> t = p;

    std::cout << "Supported HETERO config keys: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST(OVClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    // TODO: check
    std::string targetDevice = CommonTestUtils::DEVICE_HETERO + std::string(":") + CommonTestUtils::DEVICE_CPU;
    ASSERT_THROW(ie.get_metric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)), ov::Exception);
}

TEST_P(OVClassGetMetricTest_SUPPORTED_METRICS, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> t = p;

    std::cout << "Supported metrics: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> t = p;

    std::cout << "Supported config values: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(OVClassGetMetricTest_AVAILABLE_DEVICES, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)));
    std::vector<std::string> t = p;

    std::cout << "Available devices: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(AVAILABLE_DEVICES));
}

TEST_P(OVClassGetMetricTest_FULL_DEVICE_NAME, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(FULL_DEVICE_NAME)));
    std::string t = p;
    std::cout << "Full device name: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(FULL_DEVICE_NAME));
}

TEST_P(OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));
    std::vector<std::string> t = p;

    std::cout << "Optimization capabilities: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
}

TEST_P(OVClassGetMetricTest_DEVICE_GOPS, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(DEVICE_GOPS)));
    std::map<InferenceEngine::Precision, float> t = p;

    std::cout << "Device GOPS: " << std::endl;
    for (auto&& kv : t) {
        std::cout << kv.first << ": " << kv.second << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(DEVICE_GOPS));
}

TEST_P(OVClassGetMetricTest_DEVICE_TYPE, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(DEVICE_TYPE)));
    InferenceEngine::Metrics::DeviceType t = p;

    std::cout << "Device Type: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(DEVICE_TYPE));
}

TEST_P(OVClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS)));
    unsigned int t = p;

    std::cout << "Number of waiting infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS));
}

TEST_P(OVClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS)));
    unsigned int t = p;

    std::cout << "Number of executing infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS));
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)));
    std::tuple<unsigned int, unsigned int, unsigned int> t = p;

    unsigned int start = std::get<0>(t);
    unsigned int end = std::get<1>(t);
    unsigned int step = std::get<2>(t);

    std::cout << "Range for async infer requests: " << std::endl;
    std::cout << start << std::endl;
    std::cout << end << std::endl;
    std::cout << step << std::endl;
    std::cout << std::endl;

    ASSERT_LE(start, end);
    ASSERT_GE(step, 1);
    ASSERT_METRIC_SUPPORTED(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_STREAMS, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(RANGE_FOR_STREAMS)));
    std::tuple<unsigned int, unsigned int> t = p;

    unsigned int start = std::get<0>(t);
    unsigned int end = std::get<1>(t);

    std::cout << "Range for streams: " << std::endl;
    std::cout << start << std::endl;
    std::cout << end << std::endl;
    std::cout << std::endl;

    ASSERT_LE(start, end);
    ASSERT_METRIC_SUPPORTED(METRIC_KEY(RANGE_FOR_STREAMS));
}

TEST_P(OVClassGetMetricTest_ThrowUnsupported, GetMetricThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_THROW(p = ie.get_metric(deviceName, "unsupported_metric"), ov::Exception);
}

TEST_P(OVClassGetConfigTest, GetConfigNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto&& confKey : configValues) {
        ov::Any defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.get_config(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassGetConfigTest, GetConfigHeteroNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto&& confKey : configValues) {
        ASSERT_NO_THROW(ie.get_config(deviceName, confKey));
    }
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_THROW(p = ie.get_config(CommonTestUtils::DEVICE_HETERO, "unsupported_config"), ov::Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroWithDeviceThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_THROW(p = ie.get_config(CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName,
                                   HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)),
                 ov::Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_THROW(p = ie.get_config(deviceName, "unsupported_config"), ov::Exception);
}

TEST_P(OVClassSpecificDeviceTestGetConfig, GetConfigSpecificDeviceNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    std::string deviceID, clearDeviceName;
    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceName.substr(0, pos);
        deviceID =  deviceName.substr(pos + 1,  deviceName.size());
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    std::vector<std::string> deviceIDs = ie.get_metric(clearDeviceName, METRIC_KEY(AVAILABLE_DEVICES));
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_SKIP();
    }

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto &&confKey : configValues) {
        ov::Any defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.get_config(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassGetAvailableDevices, GetAvailableDevicesNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    std::vector<std::string> devices;

    ASSERT_NO_THROW(devices = ie.get_available_devices());

    bool deviceFound = false;
    std::cout << "Available devices: " << std::endl;
    for (auto&& device : devices) {
        if (device.find(deviceName) != std::string::npos) {
            deviceFound = true;
        }

        std::cout << device << " ";
    }
    std::cout << std::endl;

    ASSERT_TRUE(deviceFound);
}

//
// QueryNetwork with HETERO on particular device
//

TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithDeviceIDNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        ASSERT_NO_THROW(ie.query_model(actualNetwork,
                                       CommonTestUtils::DEVICE_HETERO,
                                       {{"TARGET_FALLBACK", deviceName + "." + deviceIDs[0] + "," + deviceName}}));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithDeviceID) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        try {
            ie.query_model(simpleNetwork, deviceName + ".0");
        } catch (const ov::Exception& ex) {
            std::string message = ex.what();
            ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
        }
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithBigDeviceIDThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork, deviceName + ".110"), ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithInvalidDeviceIDThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork, deviceName + ".l0"), ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithBigDeviceIDThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork,
                                    CommonTestUtils::DEVICE_HETERO,
                                    {{"TARGET_FALLBACK", deviceName + ".100," + deviceName}}),
                     ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

using OVClassNetworkTestP = OVClassBaseTestP;

//
// LoadNetwork
//

TEST_P(OVClassNetworkTestP, LoadNetworkActualNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, deviceName));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK", deviceName}}));
}

TEST_P(OVClassNetworkTestP, LoadNetworkCreateDefaultExecGraphResult) {
    auto ie = createCoreWithTemplate();
    auto net = ie.compile_model(actualNetwork, deviceName);
    auto runtime_function = net.get_runtime_model();
    ASSERT_NE(nullptr, runtime_function);
    auto actual_parameters = runtime_function->get_parameters();
    auto actual_results = runtime_function->get_results();
    auto expected_parameters = actualNetwork->get_parameters();
    auto expected_results = actualNetwork->get_results();
    ASSERT_EQ(expected_parameters.size(), actual_parameters.size());
    for (std::size_t i = 0; i < expected_parameters.size(); ++i) {
        auto expected_element_type = expected_parameters[i]->get_output_element_type(0);
        auto actual_element_type = actual_parameters[i]->get_output_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_parameters[i]->get_output_shape(0);
        auto actual_shape = actual_parameters[i]->get_output_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
    ASSERT_EQ(expected_results.size(), actual_results.size());
    for (std::size_t i = 0; i < expected_results.size(); ++i) {
        auto expected_element_type = expected_results[i]->get_input_element_type(0);
        auto actual_element_type = actual_results[i]->get_input_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_results[i]->get_input_shape(0);
        auto actual_shape = actual_results[i]->get_input_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
}

TEST_P(OVClassSeveralDevicesTestLoadNetwork, LoadNetworkActualSeveralDevicesNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    std::string clearDeviceName;
    auto pos = deviceNames.begin()->find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceNames.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    std::vector<std::string> deviceIDs = ie.get_metric(clearDeviceName, METRIC_KEY(AVAILABLE_DEVICES));
    if (deviceIDs.size() < deviceNames.size())
        GTEST_SKIP();

    std::string multiDeviceName = CommonTestUtils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : deviceNames) {
        multiDeviceName += dev_name;
        if (&dev_name != &(deviceNames.back())) {
            multiDeviceName += ",";
        }
    }
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, multiDeviceName));
}

//
// LoadNetwork with HETERO on particular device
//
TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithDeviceIDNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        std::string heteroDevice =
                CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName + "." + deviceIDs[0] + "," + deviceName;
        ASSERT_NO_THROW(ie.compile_model(actualNetwork, heteroDevice));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithDeviceIDNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        ASSERT_NO_THROW(ie.compile_model(simpleNetwork, deviceName + "." + deviceIDs[0]));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithBigDeviceIDThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, deviceName + ".10"), ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithInvalidDeviceIDThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, deviceName + ".l0"), ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithBigDeviceIDThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                      "HETERO",
                                      {{"TARGET_FALLBACK", deviceName + ".100," + CommonTestUtils::DEVICE_CPU}}),
                     ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROAndDeviceIDThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                      CommonTestUtils::DEVICE_HETERO,
                                      {{"TARGET_FALLBACK", deviceName + "," + CommonTestUtils::DEVICE_CPU},
                                       {CONFIG_KEY(DEVICE_ID), "110"}}),
                     ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROwithMULTINoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        std::string targetFallback(CommonTestUtils::DEVICE_MULTI + std::string(",") + deviceName);
        ASSERT_NO_THROW(
                ie.compile_model(actualNetwork,
                                 CommonTestUtils::DEVICE_HETERO,
                                 {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices}, {"TARGET_FALLBACK", targetFallback}}));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkMULTIwithHETERONoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += CommonTestUtils::DEVICE_HETERO + std::string(".") + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        ASSERT_NO_THROW(ie.compile_model(
                actualNetwork,
                CommonTestUtils::DEVICE_MULTI,
                {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices}, {"TARGET_FALLBACK", deviceName + "," + deviceName}}));
    } else {
        GTEST_SKIP();
    }
}

//
// QueryNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, QueryNetworkHETEROWithMULTINoThrow_V10) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        auto function = multinputNetwork;
        ASSERT_NE(nullptr, function);
        std::unordered_set<std::string> expectedLayers;
        for (auto&& node : function->get_ops()) {
            expectedLayers.emplace(node->get_friendly_name());
        }
        ov::runtime::SupportedOpsMap result;
        std::string targetFallback(CommonTestUtils::DEVICE_MULTI + std::string(",") + deviceName);
        ASSERT_NO_THROW(result = ie.query_model(
                multinputNetwork,
                CommonTestUtils::DEVICE_HETERO,
                {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices}, {"TARGET_FALLBACK", targetFallback}}));

        std::unordered_set<std::string> actualLayers;
        for (auto&& layer : result) {
            actualLayers.emplace(layer.first);
        }
        ASSERT_EQ(expectedLayers, actualLayers);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, QueryNetworkMULTIWithHETERONoThrow_V10) {
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += "HETERO." + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        auto function = multinputNetwork;
        ASSERT_NE(nullptr, function);
        std::unordered_set<std::string> expectedLayers;
        for (auto&& node : function->get_ops()) {
            expectedLayers.emplace(node->get_friendly_name());
        }
        ov::runtime::SupportedOpsMap result;
        ASSERT_NO_THROW(result = ie.query_model(multinputNetwork,
                                                CommonTestUtils::DEVICE_MULTI,
                                                {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                                                 {"TARGET_FALLBACK", deviceName + "," + deviceName}}));

        std::unordered_set<std::string> actualLayers;
        for (auto&& layer : result) {
            actualLayers.emplace(layer.first);
        }
        ASSERT_EQ(expectedLayers, actualLayers);
    } else {
        GTEST_SKIP();
    }
}

// TODO: Enable this test with pre-processing
TEST_P(OVClassLoadNetworkAfterCoreRecreateTest, LoadAfterRecreateCoresAndPlugins) {
    ov::runtime::Core ie = createCoreWithTemplate();
    {
        auto versions = ie.get_versions(std::string(CommonTestUtils::DEVICE_MULTI) + ":" + deviceName + "," +
                                        CommonTestUtils::DEVICE_CPU);
        ASSERT_EQ(3, versions.size());
    }
    std::map<std::string, std::string> config;
    if (deviceName == CommonTestUtils::DEVICE_CPU) {
        config.insert({"CPU_THREADS_NUM", "3"});
    }
    // ASSERT_NO_THROW({
    //     ov::runtime::Core ie = createCoreWithTemplate();
    //     std::string name = actualNetwork.getInputsInfo().begin()->first;
    //     actualNetwork.getInputsInfo().at(name)->setPrecision(Precision::U8);
    //     auto executableNetwork = ie.compile_model(actualNetwork, deviceName, config);
    // });
};

TEST_P(OVClassSetDefaultDeviceIDTest, SetDefaultDeviceIDNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    std::vector<std::string> deviceIDs = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES));
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_SKIP();
    }
    std::string value;
    ASSERT_NO_THROW(ie.set_config({{ InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, deviceID },
                                  { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES }},
                                  deviceName));
    ASSERT_NO_THROW(value = ie.get_config(deviceName, InferenceEngine::PluginConfigParams::KEY_PERF_COUNT).as<std::string>());
    ASSERT_EQ(value, InferenceEngine::PluginConfigParams::YES);
}

TEST_P(OVClassSetGlobalConfigTest, SetGlobalConfigNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();

    std::vector<std::string> deviceIDs = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES));
    ov::Any ref, src;
    for (auto& dev_id : deviceIDs) {
        ASSERT_NO_THROW(ie.set_config({{ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO }},
                                      deviceName + "." + dev_id));
    }
    ASSERT_NO_THROW(ie.set_config({{ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES }}, deviceName));
    ASSERT_NO_THROW(ref = ie.get_config(deviceName, InferenceEngine::PluginConfigParams::KEY_PERF_COUNT));

    for (auto& dev_id : deviceIDs) {
        ASSERT_NO_THROW(src = ie.get_config(deviceName + "." + dev_id, InferenceEngine::PluginConfigParams::KEY_PERF_COUNT));
        ASSERT_EQ(src, ref);
    }
}

TEST_P(OVClassSeveralDevicesTestDefaultCore, DefaultCoreSeveralDevicesNoThrow) {
    ov::runtime::Core ie;

    std::string clearDeviceName;
    auto pos = deviceNames.begin()->find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceNames.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    std::vector<std::string> deviceIDs = ie.get_metric(clearDeviceName, METRIC_KEY(AVAILABLE_DEVICES));
    if (deviceIDs.size() < deviceNames.size())
        GTEST_SKIP();

    for (size_t i = 0; i < deviceNames.size(); ++i) {
        ASSERT_NO_THROW(ie.set_config({{ InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, std::to_string(i + 2) }}, deviceNames[i]));
    }
    std::string res;
    for (size_t i = 0; i < deviceNames.size(); ++i) {
        ASSERT_NO_THROW(res = ie.get_config(deviceNames[i], InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS).as<std::string>());
        ASSERT_EQ(res, std::to_string(i + 2));
    }
}
}  // namespace behavior
}  // namespace test
}  // namespace ov

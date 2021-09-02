// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <common_test_utils/common_utils.hpp>
#include <common_test_utils/test_assertions.hpp>
#include <fstream>
#include <functional_test_utils/plugin_cache.hpp>
#include <functional_test_utils/skip_tests_config.hpp>
#include <ie_plugin_config.hpp>
#include <memory>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/variant.hpp>
#include <openvino/runtime/core.hpp>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

#ifdef ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>

#endif

using namespace testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace InferenceEngine::PluginConfigParams;

namespace BehaviorTestsDefinitions {

#define ASSERT_EXEC_METRIC_SUPPORTED(metricName)                                                \
    {                                                                                           \
        std::vector<std::string> metrics = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)); \
        auto it = std::find(metrics.begin(), metrics.end(), metricName);                        \
        ASSERT_NE(metrics.end(), it);                                                           \
    }

#define ASSERT_METRIC_SUPPORTED(metricName)                                                         \
    {                                                                                               \
        std::vector<std::string> metrics = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_METRICS)); \
        auto it = std::find(metrics.begin(), metrics.end(), metricName);                            \
        ASSERT_NE(metrics.end(), it);                                                               \
    }

#define SKIP_IF_NOT_IMPLEMENTED(...)                       \
    {                                                      \
        try {                                              \
            __VA_ARGS__;                                   \
        } catch (const InferenceEngine::NotImplemented&) { \
            GTEST_SKIP();                                  \
        }                                                  \
    }

inline ov::runtime::Core createCoreWithTemplate() {
    ov::runtime::Core ie;
    std::string pluginName = "templatePlugin";
    pluginName += IE_BUILD_POSTFIX;
    ie.register_plugin(pluginName, "TEMPLATE");
    return ie;
}

class OVClassBasicTestP : public ::testing::Test, public WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string pluginName;

public:
    void SetUp() override {
        std::tie(pluginName, deviceName) = GetParam();
        pluginName += IE_BUILD_POSTFIX;
    }
};

class OVClassNetworkTest : public ::testing::Test {
public:
    std::shared_ptr<ngraph::Function> actualNetwork, simpleNetwork, multinputNetwork, ksoNetwork;

    void SetUp() override {
        // Generic network
        {
            actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
        }
        // Quite simple network
        {
            simpleNetwork = ngraph::builder::subgraph::makeSingleConv();
        }
        // Multinput to substruct network
        {
            multinputNetwork = ngraph::builder::subgraph::make2InputSubtract();
        }
        // Network with KSO
        {
            ksoNetwork = ngraph::builder::subgraph::makeKSOFunction();
        }
    }
    void setHeteroNetworkAffinity(const std::string& targetDevice) {
        const std::map<std::string, std::string> deviceMapping = {{"Split_2", targetDevice},
                                                                  {"Convolution_4", targetDevice},
                                                                  {"Convolution_7", CommonTestUtils::DEVICE_CPU},
                                                                  {"Relu_5", CommonTestUtils::DEVICE_CPU},
                                                                  {"Relu_8", targetDevice},
                                                                  {"Concat_9", CommonTestUtils::DEVICE_CPU}};

        for (const auto& op : actualNetwork->get_ops()) {
            auto it = deviceMapping.find(op->get_friendly_name());
            if (it != deviceMapping.end()) {
                std::string affinity = it->second;
                op->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
            }
        }
    }
};

class OVClassBaseTestP : public OVClassNetworkTest, public WithParamInterface<std::string> {
public:
    std::string deviceName;
    void SetUp() override {
        OVClassNetworkTest::SetUp();
        deviceName = GetParam();
    }
};

using OVClassNetworkTestP = OVClassBaseTestP;
using OVClassGetMetricTest = OVClassBaseTestP;
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
using OVClassGetConfigTest_ThrowUnsupported = OVClassBaseTestP;
using OVClassGetConfigTest_ThrowUnsupported = OVClassBaseTestP;
using OVClassGetAvailableDevices = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest = OVClassBaseTestP;
using OVClassGetMetricTest_RANGE_FOR_STREAMS = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_NETWORK_NAME = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported = OVClassBaseTestP;
using OVClassExecutableNetworkGetConfigTest = OVClassBaseTestP;
using OVClassExecutableNetworkSetConfigTest = OVClassBaseTestP;
using OVClassExecutableNetworkGetConfigTest = OVClassBaseTestP;
using OVClassLoadNetworkAfterCoreRecreateTest = OVClassBaseTestP;

class OVClassExecutableNetworkGetMetricTestForSpecificConfig
    : public OVClassNetworkTest,
      public WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>> {
protected:
    std::string deviceName;
    std::string configKey;
    std::string configValue;

public:
    void SetUp() override {
        OVClassNetworkTest::SetUp();
        deviceName = get<0>(GetParam());
        std::tie(configKey, configValue) = get<1>(GetParam());
    }
};

using OVClassExecutableNetworkSupportedConfigTest = OVClassExecutableNetworkGetMetricTestForSpecificConfig;
using OVClassExecutableNetworkUnsupportedConfigTest = OVClassExecutableNetworkGetMetricTestForSpecificConfig;

//
// Hetero Executable network case
//
class OVClassHeteroExecutableNetworkGetMetricTest : public OVClassNetworkTest, public WithParamInterface<std::string> {
protected:
    std::string deviceName;
    std::string heteroDeviceName;

public:
    void SetUp() override {
        OVClassNetworkTest::SetUp();
        deviceName = GetParam();
        heteroDeviceName = CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName + std::string(",") +
                           CommonTestUtils::DEVICE_CPU;
    }
};
using OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassLoadNetworkTest = OVClassQueryNetworkTest;

bool supportsAvaliableDevices(ov::runtime::Core& ie, const std::string& deviceName) {
    auto supportedMetricKeys = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
    return supportedMetricKeys.end() !=
           std::find(std::begin(supportedMetricKeys), std::end(supportedMetricKeys), METRIC_KEY(AVAILABLE_DEVICES));
}

TEST(OVClassBasicTest, smoke_createDefault) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ASSERT_NO_THROW(ov::runtime::Core ie);
}

TEST_P(OVClassBasicTestP, registerExistingPluginThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.register_plugin(pluginName, deviceName), Exception);
}

TEST_P(OVClassBasicTestP, registerNewPluginNoThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.register_plugin(pluginName, "NEW_DEVICE_NAME"));
    ASSERT_NO_THROW(ie.get_metric("NEW_DEVICE_NAME", METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
}

TEST(OVClassBasicTest, smoke_registerExistingPluginFileThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.register_plugins("nonExistPlugins.xml"), Exception);
}

TEST(OVClassBasicTest, smoke_createNonExistingConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ASSERT_THROW(ov::runtime::Core ie("nonExistPlugins.xml"), Exception);
}

#ifdef __linux__

TEST(OVClassBasicTest, smoke_createMockEngineConfigNoThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::string filename{"mock_engine_valid.xml"};
    std::string content{"<ie><plugins><plugin name=\"mock\" location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_NO_THROW(ov::runtime::Core ie(filename));
    CommonTestUtils::removeFile(filename.c_str());
}

TEST(OVClassBasicTest, smoke_createMockEngineConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::string filename{"mock_engine.xml"};
    std::string content{"<ie><plugins><plugin location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_THROW(ov::runtime::Core ie(filename), Exception);
    CommonTestUtils::removeFile(filename.c_str());
}

#endif

#ifdef ENABLE_UNICODE_PATH_SUPPORT

TEST_P(OVClassBasicTestP, smoke_registerPluginsXMLUnicodePath) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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
                       << ::FileUtils::wStringtoMBCSstringChar(pluginsXmlW) << "'";
            }

            GTEST_COUT << "Test " << testIndex << std::endl;

            ov::runtime::Core ie = createCoreWithTemplate();
            GTEST_COUT << "Core created " << testIndex << std::endl;
            ASSERT_NO_THROW(ie.register_plugins(::FileUtils::wStringtoMBCSstringChar(pluginsXmlW)));
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
        } catch (const InferenceEngine::Exception& e_next) {
            CommonTestUtils::removeFile(pluginsXmlW);
            std::remove(pluginXML.c_str());
            FAIL() << e_next.what();
        }
    }
    CommonTestUtils::removeFile(pluginXML);
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

//
// GetVersions()
//

TEST_P(OVClassBasicTestP, getVersionsByExactDeviceNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.get_versions(deviceName + ".0"));
}

TEST_P(OVClassBasicTestP, getVersionsByDeviceClassNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.get_versions(deviceName));
}

TEST_P(OVClassBasicTestP, getVersionsNonEmpty) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_EQ(2, ie.get_versions(CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName).size());
}

//
// UnregisterPlugin
//

TEST_P(OVClassBasicTestP, unregisterExistingPluginNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    // device instance is not created yet
    ASSERT_THROW(ie.unload_plugin(deviceName), Exception);

    // make the first call to IE which created device instance
    ie.get_versions(deviceName);
    // now, we can unregister device
    ASSERT_NO_THROW(ie.unload_plugin(deviceName));
}

TEST_P(OVClassBasicTestP, accessToUnregisteredPluginThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.unload_plugin(deviceName), Exception);
    ASSERT_NO_THROW(ie.get_versions(deviceName));
    ASSERT_NO_THROW(ie.unload_plugin(deviceName));
    ASSERT_NO_THROW(ie.set_config({}, deviceName));
    ASSERT_NO_THROW(ie.get_versions(deviceName));
    ASSERT_NO_THROW(ie.unload_plugin(deviceName));
}

TEST(OVClassBasicTest, smoke_unregisterNonExistingPluginThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.unload_plugin("unkown_device"), Exception);
}

//
// SetConfig
//

TEST_P(OVClassBasicTestP, SetConfigAllThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{"unsupported_key", "4"}}));
    ASSERT_ANY_THROW(ie.get_versions(deviceName));
}

TEST_P(OVClassBasicTestP, SetConfigForUnRegisteredDeviceThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.set_config({{"unsupported_key", "4"}}, "unregistered_device"), Exception);
}

TEST_P(OVClassBasicTestP, SetConfigNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{KEY_PERF_COUNT, YES}}, deviceName));
}

TEST_P(OVClassBasicTestP, SetConfigAllNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{KEY_PERF_COUNT, YES}}));
    ASSERT_NO_THROW(ie.get_versions(deviceName));
}

TEST(OVClassBasicTest, smoke_SetConfigHeteroThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{KEY_PERF_COUNT, YES}}, CommonTestUtils::DEVICE_HETERO));
}

TEST_P(OVClassBasicTestP, SetConfigHeteroTargetFallbackThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.set_config({{"TARGET_FALLBACK", deviceName}}, CommonTestUtils::DEVICE_HETERO));
}

TEST(OVClassBasicTest, smoke_SetConfigHeteroNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    bool value = false;

    ASSERT_NO_THROW(ie.set_config({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), YES}}, CommonTestUtils::DEVICE_HETERO));
    ASSERT_NO_THROW(value = ie.get_config("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_TRUE(value);

    ASSERT_NO_THROW(ie.set_config({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), NO}}, CommonTestUtils::DEVICE_HETERO));
    ASSERT_NO_THROW(value = ie.get_config("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_FALSE(value);
}

//
// LoadNetwork
//

TEST_P(OVClassNetworkTestP, LoadNetworkActualNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, deviceName));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK", deviceName}}));
}

//
// ImportExportNetwork
//

TEST_P(OVClassImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.compile_model(actualNetwork, deviceName));
    ASSERT_NO_THROW(executableNetwork.Export(strm));
    ASSERT_NO_THROW(executableNetwork = ie.import_model(strm, deviceName));
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

TEST_P(OVClassImportExportTestP, smoke_ExportUsingFileNameImportFromStreamNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ExecutableNetwork executableNetwork;
    std::string fileName{"ExportedNetwork"};
    {
        ASSERT_NO_THROW(executableNetwork = ie.compile_model(simpleNetwork, deviceName));
        ASSERT_NO_THROW(executableNetwork.Export(fileName));
    }
    {
        {
            std::ifstream strm(fileName);
            ASSERT_NO_THROW(executableNetwork = ie.import_model(strm, deviceName));
        }
        ASSERT_EQ(0, remove(fileName.c_str()));
    }
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

//
// QueryNetwork
//

TEST_P(OVClassNetworkTestP, QueryNetworkActualThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.query_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(OVClassNetworkTestP, QueryNetworkActualNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();

    try {
        ie.query_model(actualNetwork, deviceName);
    } catch (const InferenceEngine::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassNetworkTestP, QueryNetworkWithKSO) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();

    try {
        auto rres = ie.query_model(ksoNetwork, deviceName);
        auto rl_map = rres.supportedLayersMap;
        auto func = ksoNetwork;
        for (const auto& op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
    } catch (const InferenceEngine::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassNetworkTestP, SetAffinityWithConstantBranches) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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

        auto rres = ie.query_model(func, deviceName);
        auto rl_map = rres.supportedLayersMap;
        for (const auto& op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
        for (const auto& op : func->get_ops()) {
            std::string affinity = rl_map[op->get_friendly_name()];
            op->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
        }
        ExecutableNetwork exeNetwork = ie.compile_model(ksoNetwork, deviceName);
    } catch (const NotImplemented& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassNetworkTestP, SetAffinityWithKSO) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();

    try {
        auto rres = ie.query_model(ksoNetwork, deviceName);
        auto rl_map = rres.supportedLayersMap;
        auto func = ksoNetwork;
        for (const auto& op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
        for (const auto& op : func->get_ops()) {
            std::string affinity = rl_map[op->get_friendly_name()];
            op->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
        }
        ExecutableNetwork exeNetwork = ie.compile_model(ksoNetwork, deviceName);
    } catch (const InferenceEngine::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassNetworkTestP, QueryNetworkHeteroActualNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    QueryNetworkResult res;
    ASSERT_NO_THROW(
        res = ie.query_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK", deviceName}}));
    ASSERT_LT(0, res.supportedLayersMap.size());
}

TEST_P(OVClassNetworkTestP, QueryNetworkMultiThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.query_model(actualNetwork, CommonTestUtils::DEVICE_MULTI), Exception);
}

TEST(OVClassBasicTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    // TODO: check
    std::string targetDevice = CommonTestUtils::DEVICE_HETERO + std::string(":") + CommonTestUtils::DEVICE_CPU;
    ASSERT_THROW(ie.get_metric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)), Exception);
}

TEST_P(OVClassGetMetricTest_SUPPORTED_METRICS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> t = p;

    std::cout << "Supported metrics: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> t = p;

    std::cout << "Supported config values: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(OVClassGetMetricTest_AVAILABLE_DEVICES, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)));
    std::vector<std::string> t = p;

    std::cout << "Available devices: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(AVAILABLE_DEVICES));
}

TEST_P(OVClassGetMetricTest_FULL_DEVICE_NAME, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(FULL_DEVICE_NAME)));
    std::string t = p;
    std::cout << "Full device name: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(FULL_DEVICE_NAME));
}

TEST_P(OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));
    std::vector<std::string> t = p;

    std::cout << "Optimization capabilities: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
}

TEST_P(OVClassGetMetricTest_DEVICE_GOPS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(DEVICE_GOPS)));
    std::map<InferenceEngine::Precision, float> t = p;

    std::cout << "Device GOPS: " << std::endl;
    for (auto&& kv : t) {
        std::cout << kv.first << ": " << kv.second << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(DEVICE_GOPS));
}

TEST_P(OVClassGetMetricTest_DEVICE_TYPE, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(DEVICE_TYPE)));
    InferenceEngine::Metrics::DeviceType t = p;

    std::cout << "Device Type: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(DEVICE_TYPE));
}

TEST_P(OVClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS)));
    unsigned int t = p;

    std::cout << "Number of waiting infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS));
}

TEST_P(OVClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS)));
    unsigned int t = p;

    std::cout << "Number of executing infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS));
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_THROW(p = ie.get_metric(deviceName, "unsupported_metric"), Exception);
}

TEST_P(OVClassGetConfigTest, GetConfigNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto&& confKey : configValues) {
        Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.get_config(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassGetConfigTest, GetConfigHeteroNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto&& confKey : configValues) {
        ASSERT_NO_THROW(ie.get_config(deviceName, confKey));
    }
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_THROW(p = ie.get_config(CommonTestUtils::DEVICE_HETERO, "unsupported_config"), Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroWithDeviceThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_THROW(p = ie.get_config(CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName,
                                  HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)),
                 Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_THROW(p = ie.get_config(deviceName, "unsupported_config"), Exception);
}

TEST_P(OVClassGetAvailableDevices, GetAvailableDevicesNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
// ExecutableNetwork GetMetric / GetConfig
//
TEST_P(OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    std::cout << "Supported config keys: " << std::endl;
    for (auto&& conf : configValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, configValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> metricValues = p;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto&& conf : metricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, metricValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
    ASSERT_EQ(simpleNetwork->get_friendly_name(), networkname);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)));
    unsigned int value = p;

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.GetMetric("unsupported_metric"), Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto&& confKey : configValues) {
        Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.get_config(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.GetConfig("unsupported_config"), Exception);
}

TEST_P(OVClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.SetConfig({{"unsupported_config", "some_value"}}), Exception);
}

TEST_P(OVClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(exeNetwork.SetConfig({{configKey, configValue}}));
    ASSERT_NO_THROW(p = exeNetwork.GetConfig(configKey));
    ASSERT_EQ(p, configValue);
}

TEST_P(OVClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.SetConfig({{configKey, configValue}}), Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> devConfigValues = p;

    ExecutableNetwork exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> execConfigValues = p;

    /*
    for (auto && configKey : devConfigValues) {
        ASSERT_NE(execConfigValues.end(), std::find(execConfigValues.begin(), execConfigValues.end(), configKey));

        Parameter configValue;
        ASSERT_NO_THROW(Parameter configValue = exeNetwork.get_config(configKey));
    }
    */
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter pHetero, pDevice;

    ExecutableNetwork heteroExeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);
    ExecutableNetwork deviceExeNetwork = ie.compile_model(actualNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> heteroConfigValues = pHetero, deviceConfigValues = pDevice;

    std::cout << "Supported config keys: " << std::endl;
    for (auto&& conf : heteroConfigValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, heteroConfigValues.size());

    // check that all device config values are present in hetero case
    for (auto&& deviceConf : deviceConfigValues) {
        auto it = std::find(heteroConfigValues.begin(), heteroConfigValues.end(), deviceConf);
        ASSERT_TRUE(it != heteroConfigValues.end());

        Parameter heteroConfigValue = heteroExeNetwork.GetConfig(deviceConf);
        Parameter deviceConfigValue = deviceExeNetwork.GetConfig(deviceConf);

        // HETERO returns EXCLUSIVE_ASYNC_REQUESTS as a boolean value
        if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) != deviceConf) {
            ASSERT_EQ(deviceConfigValue, heteroConfigValue);
        }
    }
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter pHetero, pDevice;

    ExecutableNetwork heteroExeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);
    ExecutableNetwork deviceExeNetwork = ie.compile_model(actualNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> heteroMetricValues = pHetero, deviceMetricValues = pDevice;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto&& conf : heteroMetricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, heteroMetricValues.size());

    const std::vector<std::string> heteroSpecificMetrics = {METRIC_KEY(SUPPORTED_METRICS),
                                                            METRIC_KEY(SUPPORTED_CONFIG_KEYS)};

    // check that all device metric values are present in hetero case
    for (auto&& deviceMetricName : deviceMetricValues) {
        auto it = std::find(heteroMetricValues.begin(), heteroMetricValues.end(), deviceMetricName);
        ASSERT_TRUE(it != heteroMetricValues.end());

        Parameter heteroMetricValue = heteroExeNetwork.GetMetric(deviceMetricName);
        Parameter deviceMetricValue = deviceExeNetwork.GetMetric(deviceMetricName);

        if (std::find(heteroSpecificMetrics.begin(), heteroSpecificMetrics.end(), deviceMetricName) ==
            heteroSpecificMetrics.end()) {
            ASSERT_TRUE(heteroMetricValue == deviceMetricValue);
        }
    }
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();
    Parameter p;

    setHeteroNetworkAffinity(deviceName);

    ExecutableNetwork exeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetConfig("TARGET_FALLBACK"));
    std::string targets = p;
    auto expectedTargets = deviceName + "," + CommonTestUtils::DEVICE_CPU;

    std::cout << "Exe network fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}

//
// QueryNetwork with HETERO on particular device
//
bool supportsDeviceID(ov::runtime::Core& ie, const std::string& deviceName) {
    auto supportedConfigKeys =
        ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
    return supportedConfigKeys.end() !=
           std::find(std::begin(supportedConfigKeys), std::end(supportedConfigKeys), CONFIG_KEY(DEVICE_ID));
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithDeviceIDNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        try {
            ie.query_model(simpleNetwork, deviceName + ".0");
        } catch (const InferenceEngine::Exception& ex) {
            std::string message = ex.what();
            ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
        }
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithBigDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork, deviceName + ".110"), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithInvalidDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork, deviceName + ".l0"), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithBigDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork,
                                     CommonTestUtils::DEVICE_HETERO,
                                     {{"TARGET_FALLBACK", deviceName + ".100," + deviceName}}),
                     Exception);
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with HETERO on particular device
//
TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithDeviceIDNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, deviceName + ".10"), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithInvalidDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, deviceName + ".l0"), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithBigDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                    "HETERO",
                                    {{"TARGET_FALLBACK", deviceName + ".100," + CommonTestUtils::DEVICE_CPU}}),
                     Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROAndDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::runtime::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                    CommonTestUtils::DEVICE_HETERO,
                                    {{"TARGET_FALLBACK", deviceName + "," + CommonTestUtils::DEVICE_CPU},
                                     {CONFIG_KEY(DEVICE_ID), "110"}}),
                     Exception);
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROwithMULTINoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
        QueryNetworkResult result;
        std::string targetFallback(CommonTestUtils::DEVICE_MULTI + std::string(",") + deviceName);
        ASSERT_NO_THROW(result = ie.query_model(
                            multinputNetwork,
                            CommonTestUtils::DEVICE_HETERO,
                            {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices}, {"TARGET_FALLBACK", targetFallback}}));

        std::unordered_set<std::string> actualLayers;
        for (auto&& layer : result.supportedLayersMap) {
            actualLayers.emplace(layer.first);
        }
        ASSERT_EQ(expectedLayers, actualLayers);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, QueryNetworkMULTIWithHETERONoThrow_V10) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
        QueryNetworkResult result;
        ASSERT_NO_THROW(result = ie.query_model(multinputNetwork,
                                                 CommonTestUtils::DEVICE_MULTI,
                                                 {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                                                  {"TARGET_FALLBACK", deviceName + "," + deviceName}}));

        std::unordered_set<std::string> actualLayers;
        for (auto&& layer : result.supportedLayersMap) {
            actualLayers.emplace(layer.first);
        }
        ASSERT_EQ(expectedLayers, actualLayers);
    } else {
        GTEST_SKIP();
    }
}

// TODO: Enable this test with pre-processing
TEST_P(OVClassLoadNetworkAfterCoreRecreateTest, DISABLED_LoadAfterRecreateCoresAndPlugins) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
}  // namespace BehaviorTestsDefinitions

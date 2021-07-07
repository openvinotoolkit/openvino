// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <memory>
#include <fstream>
#include <ngraph/variant.hpp>
#include <functional_test_utils/plugin_cache.hpp>
#include <ngraph/op/util/op_types.hpp>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

#include <functional_test_utils/skip_tests_config.hpp>
#include <common_test_utils/common_utils.hpp>
#include <common_test_utils/test_assertions.hpp>

#ifdef ENABLE_UNICODE_PATH_SUPPORT
#include <iostream>
#define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#include <codecvt>
#include <functional_test_utils/skip_tests_config.hpp>

#endif

using namespace testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace InferenceEngine::PluginConfigParams;

namespace BehaviorTestsDefinitions {                                                    \

#define ASSERT_EXEC_METRIC_SUPPORTED(metricName)                     \
{                                                                    \
    std::vector<std::string> metrics =                               \
        exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS));         \
    auto it = std::find(metrics.begin(), metrics.end(), metricName); \
    ASSERT_NE(metrics.end(), it);                                    \
}

#define ASSERT_METRIC_SUPPORTED(metricName)                          \
{                                                                    \
    std::vector<std::string> metrics =                               \
        ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));     \
    auto it = std::find(metrics.begin(), metrics.end(), metricName); \
    ASSERT_NE(metrics.end(), it);                                    \
}

#define SKIP_IF_NOT_IMPLEMENTED(...)                                            \
{                                                                               \
    try {                                                                       \
        __VA_ARGS__;                                                            \
    } catch (const InferenceEngine::NotImplemented&) {                          \
        GTEST_SKIP();                                                           \
    }                                                                           \
}


class IEClassBasicTestP : public ::testing::Test, public WithParamInterface<std::pair<std::string, std::string> > {
protected:
    std::string deviceName;
    std::string pluginName;
public:
    void SetUp() override {
        std::tie(pluginName, deviceName) = GetParam();
        pluginName += IE_BUILD_POSTFIX;
    }
};

class IEClassNetworkTest : public ::testing::Test {
public:
    CNNNetwork actualNetwork, simpleNetwork, multinputNetwork, ksoNetwork;

    void SetUp() override {
        // Generic network
        {
            std::shared_ptr<ngraph::Function> fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
            ASSERT_NO_THROW(actualNetwork = CNNNetwork(fnPtr));
        }
        // Quite simple network
        {
            std::shared_ptr<ngraph::Function> fnPtr = ngraph::builder::subgraph::makeSingleConv();
            ASSERT_NO_THROW(simpleNetwork = CNNNetwork(fnPtr));
        }
        // Multinput to substruct network
        {
            auto fnPtr = ngraph::builder::subgraph::make2InputSubtract();
            multinputNetwork = InferenceEngine::CNNNetwork(fnPtr);
        }
        // Network with KSO
        {
            auto fnPtr = ngraph::builder::subgraph::makeKSOFunction();
            ksoNetwork = InferenceEngine::CNNNetwork(fnPtr);
        }
    }
    void setHeteroNetworkAffinity(const std::string& targetDevice) {
        const std::map<std::string, std::string> deviceMapping = {
                {"Split_2",         targetDevice},
                {"Convolution_4",   targetDevice},
                {"Convolution_7",   CommonTestUtils::DEVICE_CPU},
                {"Relu_5",          CommonTestUtils::DEVICE_CPU},
                {"Relu_8",          targetDevice},
                {"Concat_9",        CommonTestUtils::DEVICE_CPU}
        };

        for (const auto & op : actualNetwork.getFunction()->get_ops()) {
            auto it = deviceMapping.find(op->get_friendly_name());
            if (it != deviceMapping.end()) {
                std::string affinity = it->second;
                op->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
            }
        }
    }
};

class IEClassBaseTestP : public IEClassNetworkTest, public WithParamInterface<std::string> {
public:
    std::string deviceName;
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        deviceName = GetParam();
    }
};

using IEClassNetworkTestP = IEClassBaseTestP;
using IEClassGetMetricTest = IEClassBaseTestP;
using IEClassQueryNetworkTest = IEClassBaseTestP;
using IEClassImportExportTestP = IEClassBaseTestP;
using IEClassGetMetricTest_SUPPORTED_METRICS = IEClassBaseTestP;
using IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassBaseTestP;
using IEClassGetMetricTest_AVAILABLE_DEVICES = IEClassBaseTestP;
using IEClassGetMetricTest_FULL_DEVICE_NAME = IEClassBaseTestP;
using IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES = IEClassBaseTestP;
using IEClassGetMetricTest_DEVICE_GOPS = IEClassBaseTestP;
using IEClassGetMetricTest_DEVICE_TYPE = IEClassBaseTestP;
using IEClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS = IEClassBaseTestP;
using IEClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS = IEClassBaseTestP;
using IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS = IEClassBaseTestP;
using IEClassGetMetricTest_ThrowUnsupported = IEClassBaseTestP;
using IEClassGetConfigTest = IEClassBaseTestP;
using IEClassGetConfigTest_ThrowUnsupported = IEClassBaseTestP;
using IEClassGetConfigTest_ThrowUnsupported = IEClassBaseTestP;
using IEClassGetConfigTest_ThrowUnsupported = IEClassBaseTestP;
using IEClassGetAvailableDevices = IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest = IEClassBaseTestP;
using IEClassGetMetricTest_RANGE_FOR_STREAMS = IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_NETWORK_NAME = IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported = IEClassBaseTestP;
using IEClassExecutableNetworkGetConfigTest = IEClassBaseTestP;
using IEClassExecutableNetworkSetConfigTest = IEClassBaseTestP;
using IEClassExecutableNetworkGetConfigTest = IEClassBaseTestP;
using IEClassLoadNetworkAfterCoreRecreateTest = IEClassBaseTestP;

class IEClassExecutableNetworkGetMetricTestForSpecificConfig : public IEClassNetworkTest,
                                                               public WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>> {
protected:
    std::string deviceName;
    std::string configKey;
    std::string configValue;
public:
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        deviceName = get<0>(GetParam());
        std::tie(configKey, configValue) = get<1>(GetParam());
    }
};

using IEClassExecutableNetworkSupportedConfigTest = IEClassExecutableNetworkGetMetricTestForSpecificConfig;
using IEClassExecutableNetworkUnsupportedConfigTest = IEClassExecutableNetworkGetMetricTestForSpecificConfig;

//
// Hetero Executable network case
//
class IEClassHeteroExecutableNetworkGetMetricTest : public IEClassNetworkTest, public WithParamInterface<std::string> {
protected:
    std::string deviceName;
    std::string heteroDeviceName;
public:
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        deviceName = GetParam();
        heteroDeviceName = CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName + std::string(",") + CommonTestUtils::DEVICE_CPU;
    }
};
using IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassLoadNetworkTest = IEClassQueryNetworkTest;

bool supportsAvaliableDevices(Core &ie, const std::string &deviceName) {
    auto supportedMetricKeys = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
    return supportedMetricKeys.end() != std::find(std::begin(supportedMetricKeys),
                                                  std::end(supportedMetricKeys),
                                                  METRIC_KEY(AVAILABLE_DEVICES));
}

TEST(IEClassBasicTest, smoke_createDefault) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ASSERT_NO_THROW(Core ie);
}

TEST_P(IEClassBasicTestP, registerExistingPluginThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_THROW(ie.RegisterPlugin(pluginName, deviceName), Exception);
}

TEST_P(IEClassBasicTestP, registerNewPluginNoThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.RegisterPlugin(pluginName, "NEW_DEVICE_NAME"));
    ASSERT_NO_THROW(ie.GetMetric("NEW_DEVICE_NAME", METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
}

TEST(IEClassBasicTest, smoke_registerExistingPluginFileThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_THROW(ie.RegisterPlugins("nonExistPlugins.xml"), Exception);
}

TEST(IEClassBasicTest, smoke_createNonExistingConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ASSERT_THROW(Core ie("nonExistPlugins.xml"), Exception);
}

#ifdef __linux__

TEST(IEClassBasicTest, smoke_createMockEngineConfigNoThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::string filename{"mock_engine_valid.xml"};
    std::string content{"<ie><plugins><plugin name=\"mock\" location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_NO_THROW(Core ie(filename));
    CommonTestUtils::removeFile(filename.c_str());
}

TEST(IEClassBasicTest, smoke_createMockEngineConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::string filename{"mock_engine.xml"};
    std::string content{"<ie><plugins><plugin location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_THROW(Core ie(filename), Exception);
    CommonTestUtils::removeFile(filename.c_str());
}

#endif

#ifdef ENABLE_UNICODE_PATH_SUPPORT

TEST_P(IEClassBasicTestP, smoke_registerPluginsXMLUnicodePath) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::string pluginXML{"mock_engine_valid.xml"};
    std::string content{"<ie><plugins><plugin name=\"mock\" location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(pluginXML, content);

    for (std::size_t testIndex = 0; testIndex < CommonTestUtils::test_unicode_postfix_vector.size(); testIndex++) {
        GTEST_COUT << testIndex;
        std::wstring postfix  = L"_" + CommonTestUtils::test_unicode_postfix_vector[testIndex];
        std::wstring pluginsXmlW = CommonTestUtils::addUnicodePostfixToPath(pluginXML, postfix);

        try {
            bool is_copy_successfully;
            is_copy_successfully = CommonTestUtils::copyFile(pluginXML, pluginsXmlW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << pluginXML << "' to '" << ::FileUtils::wStringtoMBCSstringChar(pluginsXmlW) << "'";
            }

            GTEST_COUT << "Test " << testIndex << std::endl;

            Core ie;
            GTEST_COUT << "Core created " << testIndex << std::endl;
            ASSERT_NO_THROW(ie.RegisterPlugins(::FileUtils::wStringtoMBCSstringChar(pluginsXmlW)));
            CommonTestUtils::removeFile(pluginsXmlW);
#if defined __linux__  && !defined(__APPLE__)
            ASSERT_NO_THROW(ie.GetVersions("mock")); // from pluginXML
#endif
            ASSERT_NO_THROW(ie.GetVersions(deviceName));
            GTEST_COUT << "Plugin created " << testIndex << std::endl;

            ASSERT_NO_THROW(ie.RegisterPlugin(pluginName, "TEST_DEVICE"));
            ASSERT_NO_THROW(ie.GetVersions("TEST_DEVICE"));
            GTEST_COUT << "Plugin registered and created " << testIndex << std::endl;

            GTEST_COUT << "OK" << std::endl;
        }
        catch (const InferenceEngine::Exception &e_next) {
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

TEST_P(IEClassBasicTestP, getVersionsByExactDeviceNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.GetVersions(deviceName + ".0"));
}

TEST_P(IEClassBasicTestP, getVersionsByDeviceClassNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.GetVersions(deviceName));
}

TEST_P(IEClassBasicTestP, getVersionsNonEmpty) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_EQ(2, ie.GetVersions(CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName).size());
}

//
// UnregisterPlugin
//

TEST_P(IEClassBasicTestP, unregisterExistingPluginNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    // device instance is not created yet
    ASSERT_THROW(ie.UnregisterPlugin(deviceName), Exception);

    // make the first call to IE which created device instance
    ie.GetVersions(deviceName);
    // now, we can unregister device
    ASSERT_NO_THROW(ie.UnregisterPlugin(deviceName));
}

TEST_P(IEClassBasicTestP, accessToUnregisteredPluginThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_THROW(ie.UnregisterPlugin(deviceName), Exception);
    ASSERT_NO_THROW(ie.GetVersions(deviceName));
    ASSERT_NO_THROW(ie.UnregisterPlugin(deviceName));
    ASSERT_NO_THROW(ie.SetConfig({}, deviceName));
    ASSERT_NO_THROW(ie.GetVersions(deviceName));
    ASSERT_NO_THROW(ie.UnregisterPlugin(deviceName));
}

TEST(IEClassBasicTest, smoke_unregisterNonExistingPluginThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_THROW(ie.UnregisterPlugin("unkown_device"), Exception);
}

//
// SetConfig
//

TEST_P(IEClassBasicTestP, SetConfigAllThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({{"unsupported_key", "4"}}));
    ASSERT_ANY_THROW(ie.GetVersions(deviceName));
}

TEST_P(IEClassBasicTestP, SetConfigForUnRegisteredDeviceThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_THROW(ie.SetConfig({{"unsupported_key", "4"}}, "unregistered_device"), Exception);
}

TEST_P(IEClassBasicTestP, SetConfigNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({{KEY_PERF_COUNT, YES}}, deviceName));
}

TEST_P(IEClassBasicTestP, SetConfigAllNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({{KEY_PERF_COUNT, YES}}));
    ASSERT_NO_THROW(ie.GetVersions(deviceName));
}

TEST(IEClassBasicTest, smoke_SetConfigHeteroThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({{KEY_PERF_COUNT, YES}}, CommonTestUtils::DEVICE_HETERO));
}

TEST_P(IEClassBasicTestP, SetConfigHeteroTargetFallbackThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({{"TARGET_FALLBACK", deviceName}}, CommonTestUtils::DEVICE_HETERO));
}

TEST(IEClassBasicTest, smoke_SetConfigHeteroNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    bool value = false;

    ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), YES}}, CommonTestUtils::DEVICE_HETERO));
    ASSERT_NO_THROW(value = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_TRUE(value);

    ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), NO}}, CommonTestUtils::DEVICE_HETERO));
    ASSERT_NO_THROW(value = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_FALSE(value);
}

//
// ImportNetwork
//

TEST_P(IEClassBasicTestP, ImportNetworkThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;

    if (deviceName == CommonTestUtils::DEVICE_CPU ||
        deviceName == CommonTestUtils::DEVICE_GPU) {
        ASSERT_THROW(ie.ImportNetwork("model", deviceName), NetworkNotRead);

        const std::string modelName = "compiled_blob.blob";
        {
            std::ofstream file(modelName);
            file << "content";
        }

        EXPECT_THROW(ie.ImportNetwork(modelName, deviceName), NotImplemented);
        ASSERT_EQ(0, std::remove(modelName.c_str()));
    }
}

TEST(IEClassBasicTest, smoke_ImportNetworkHeteroThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;

    ASSERT_THROW(ie.ImportNetwork("model", CommonTestUtils::DEVICE_HETERO), NetworkNotRead);
}

TEST(IEClassBasicTest, smoke_ImportNetworkMultiThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    ASSERT_THROW(ie.ImportNetwork("model", CommonTestUtils::DEVICE_MULTI), NetworkNotRead);
}

TEST_P(IEClassBasicTestP, ImportNetworkWithNullContextThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    RemoteContext::Ptr context = nullptr;
    std::istringstream stream("None");
    ASSERT_THROW(ie.ImportNetwork(stream, context, {}), Exception);
}

//
// LoadNetwork
//

TEST_P(IEClassNetworkTestP, LoadNetworkActualNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, deviceName));
}

TEST_P(IEClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(IEClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK", deviceName}}));
}

//
// ImportExportNetwork
//

TEST_P(IEClassImportExportTestP, smoke_ImportNetworkThrowsIfNoDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
    ASSERT_NO_THROW(executableNetwork.Export(strm));

    IE_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(executableNetwork = ie.ImportNetwork(strm), Exception);
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(IEClassImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
    ASSERT_NO_THROW(executableNetwork.Export(strm));
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(strm, deviceName));
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

TEST_P(IEClassImportExportTestP, smoke_ExportUsingFileNameImportFromStreamNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ExecutableNetwork executableNetwork;
    std::string fileName{"ExportedNetwork"};
    {
        ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(simpleNetwork, deviceName));
        ASSERT_NO_THROW(executableNetwork.Export(fileName));
    }
    {
        {
            std::ifstream strm(fileName);
            ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(strm, deviceName));
        }
        ASSERT_EQ(0, remove(fileName.c_str()));
    }
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

//
// QueryNetwork
//

TEST_P(IEClassNetworkTestP, QueryNetworkActualThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_NO_THROW(ie.QueryNetwork(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(IEClassNetworkTestP, QueryNetworkActualNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;

    try {
        ie.QueryNetwork(actualNetwork, deviceName);
    } catch (const InferenceEngine::Exception & ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(IEClassNetworkTestP, QueryNetworkWithKSO) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;

    try {
        auto rres = ie.QueryNetwork(ksoNetwork, deviceName);
        auto rl_map = rres.supportedLayersMap;
        auto func = ksoNetwork.getFunction();
        for (const auto & op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
    } catch (const InferenceEngine::Exception & ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(IEClassNetworkTestP, SetAffinityWithConstantBranches) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;

    try {
        std::shared_ptr<ngraph::Function> func;
        {
            ngraph::PartialShape shape({1, 84});
            ngraph::element::Type type(ngraph::element::Type_t::f32);
            auto param = std::make_shared<ngraph::opset6::Parameter>(type, shape);
            auto matMulWeights =
                ngraph::opset6::Constant::create(ngraph::element::Type_t::f32, {10, 84}, {1});
            auto shapeOf = std::make_shared<ngraph::opset6::ShapeOf>(matMulWeights);
            auto gConst1 = ngraph::opset6::Constant::create(ngraph::element::Type_t::i32, {1}, {1});
            auto gConst2 = ngraph::opset6::Constant::create(ngraph::element::Type_t::i64, {}, {0});
            auto gather = std::make_shared<ngraph::opset6::Gather>(shapeOf, gConst1, gConst2);
            auto concatConst = ngraph::opset6::Constant::create(ngraph::element::Type_t::i64, {1}, {1});
            auto concat =
                std::make_shared<ngraph::opset6::Concat>(ngraph::NodeVector{concatConst, gather}, 0);
            auto relu = std::make_shared<ngraph::opset6::Relu>(param);
            auto reshape = std::make_shared<ngraph::opset6::Reshape>(relu, concat, false);
            auto matMul = std::make_shared<ngraph::opset6::MatMul>(reshape, matMulWeights, false, true);
            auto matMulBias =
                ngraph::opset6::Constant::create(ngraph::element::Type_t::f32, {1, 10}, {1});
            auto addBias = std::make_shared<ngraph::opset6::Add>(matMul, matMulBias);
            auto result = std::make_shared<ngraph::opset6::Result>(addBias);

            ngraph::ParameterVector params = {param};
            ngraph::ResultVector results = {result};

            func = std::make_shared<ngraph::Function>(results, params);
        }
        CNNNetwork net(func);

        auto rres = ie.QueryNetwork(net, deviceName);
        auto rl_map = rres.supportedLayersMap;
        for (const auto & op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
        for (const auto & op : net.getFunction()->get_ops()) {
            std::string affinity = rl_map[op->get_friendly_name()];
            op->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
        }
        ExecutableNetwork exeNetwork = ie.LoadNetwork(ksoNetwork, deviceName);
    } catch (const NotImplemented& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(IEClassNetworkTestP, SetAffinityWithKSO) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;

    try {
        auto rres = ie.QueryNetwork(ksoNetwork, deviceName);
        auto rl_map = rres.supportedLayersMap;
        auto func = ksoNetwork.getFunction();
        for (const auto & op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << deviceName;
            }
        }
        for (const auto & op : ksoNetwork.getFunction()->get_ops()) {
            std::string affinity = rl_map[op->get_friendly_name()];
            op->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
        }
        ExecutableNetwork exeNetwork = ie.LoadNetwork(ksoNetwork, deviceName);
    } catch (const InferenceEngine::Exception & ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(IEClassNetworkTestP, QueryNetworkHeteroActualNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    QueryNetworkResult res;
    ASSERT_NO_THROW(res = ie.QueryNetwork(actualNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK", deviceName}}));
    ASSERT_LT(0, res.supportedLayersMap.size());
}

TEST_P(IEClassNetworkTestP, QueryNetworkMultiThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    ASSERT_THROW(ie.QueryNetwork(actualNetwork, CommonTestUtils::DEVICE_MULTI), Exception);
}

TEST(IEClassBasicTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;
    std::string deviceName = CommonTestUtils::DEVICE_HETERO;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> t = p;

    std::cout << "Supported HETERO metrics: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST(IEClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;
    std::string deviceName = CommonTestUtils::DEVICE_HETERO;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> t = p;

    std::cout << "Supported HETERO config keys: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST(IEClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    // TODO: check
    std::string targetDevice = CommonTestUtils::DEVICE_HETERO + std::string(":") + CommonTestUtils::DEVICE_CPU;
    ASSERT_THROW(ie.GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)), Exception);
}

TEST_P(IEClassGetMetricTest_SUPPORTED_METRICS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> t = p;

    std::cout << "Supported metrics: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> t = p;

    std::cout << "Supported config values: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(IEClassGetMetricTest_AVAILABLE_DEVICES, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)));
    std::vector<std::string> t = p;

    std::cout << "Available devices: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(AVAILABLE_DEVICES));
}

TEST_P(IEClassGetMetricTest_FULL_DEVICE_NAME, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(FULL_DEVICE_NAME)));
    std::string t = p;
    std::cout << "Full device name: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(FULL_DEVICE_NAME));
}

TEST_P(IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));
    std::vector<std::string> t = p;

    std::cout << "Optimization capabilities: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
}

TEST_P(IEClassGetMetricTest_DEVICE_GOPS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(DEVICE_GOPS)));
    std::map<InferenceEngine::Precision, float> t = p;

    std::cout << "Device GOPS: " << std::endl;
    for (auto &&kv : t) {
        std::cout << kv.first << ": " << kv.second << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(DEVICE_GOPS));
}

TEST_P(IEClassGetMetricTest_DEVICE_TYPE, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(DEVICE_TYPE)));
    InferenceEngine::Metrics::DeviceType t = p;

    std::cout << "Device Type: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(DEVICE_TYPE));
}

TEST_P(IEClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS)));
    unsigned int t = p;

    std::cout << "Number of waiting infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS));
}

TEST_P(IEClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS)));
    unsigned int t = p;

    std::cout << "Number of executing infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS));
}

TEST_P(IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)));
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

TEST_P(IEClassGetMetricTest_RANGE_FOR_STREAMS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(RANGE_FOR_STREAMS)));
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

TEST_P(IEClassGetMetricTest_ThrowUnsupported, GetMetricThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_THROW(p = ie.GetMetric(deviceName, "unsupported_metric"), Exception);
}

TEST_P(IEClassGetConfigTest, GetConfigNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto &&confKey : configValues) {
        Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.GetConfig(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(IEClassGetConfigTest, GetConfigHeteroNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto &&confKey : configValues) {
        ASSERT_NO_THROW(ie.GetConfig(deviceName, confKey));
    }
}

TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigHeteroThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_THROW(p = ie.GetConfig(CommonTestUtils::DEVICE_HETERO, "unsupported_config"), Exception);
}

TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigHeteroWithDeviceThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_THROW(p = ie.GetConfig(CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName, HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)),
                 Exception);
}

TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_THROW(p = ie.GetConfig(deviceName, "unsupported_config"), Exception);
}

TEST_P(IEClassGetAvailableDevices, GetAvailableDevicesNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    std::vector<std::string> devices;

    ASSERT_NO_THROW(devices = ie.GetAvailableDevices());

    bool deviceFound = false;
    std::cout << "Available devices: " << std::endl;
    for (auto &&device : devices) {
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
TEST_P(IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    std::cout << "Supported config keys: " << std::endl;
    for (auto &&conf : configValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, configValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> metricValues = p;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto &&conf : metricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, metricValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
    ASSERT_EQ(simpleNetwork.getName(), networkname);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)));
    unsigned int value = p;

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.GetMetric("unsupported_metric"), Exception);
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto &&confKey : configValues) {
        Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.GetConfig(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.GetConfig("unsupported_config"), Exception);
}

TEST_P(IEClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.SetConfig({{"unsupported_config", "some_value"}}), Exception);
}

TEST_P(IEClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(exeNetwork.SetConfig({{configKey, configValue}}));
    ASSERT_NO_THROW(p = exeNetwork.GetConfig(configKey));
    ASSERT_EQ(p, configValue);
}


TEST_P(IEClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.SetConfig({{configKey, configValue}}), Exception);
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> devConfigValues = p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> execConfigValues = p;

    /*
    for (auto && configKey : devConfigValues) {
        ASSERT_NE(execConfigValues.end(), std::find(execConfigValues.begin(), execConfigValues.end(), configKey));

        Parameter configValue;
        ASSERT_NO_THROW(Parameter configValue = exeNetwork.GetConfig(configKey));
    }
    */
}

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter pHetero, pDevice;

    ExecutableNetwork heteroExeNetwork = ie.LoadNetwork(actualNetwork, heteroDeviceName);
    ExecutableNetwork deviceExeNetwork = ie.LoadNetwork(actualNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> heteroConfigValues = pHetero, deviceConfigValues = pDevice;

    std::cout << "Supported config keys: " << std::endl;
    for (auto &&conf : heteroConfigValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, heteroConfigValues.size());

    // check that all device config values are present in hetero case
    for (auto &&deviceConf : deviceConfigValues) {
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

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter pHetero, pDevice;

    ExecutableNetwork heteroExeNetwork = ie.LoadNetwork(actualNetwork, heteroDeviceName);
    ExecutableNetwork deviceExeNetwork = ie.LoadNetwork(actualNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> heteroMetricValues = pHetero, deviceMetricValues = pDevice;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto &&conf : heteroMetricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, heteroMetricValues.size());

    const std::vector<std::string> heteroSpecificMetrics = {
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
    };

    // check that all device metric values are present in hetero case
    for (auto &&deviceMetricName : deviceMetricValues) {
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


TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
}

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    Parameter p;

    setHeteroNetworkAffinity(deviceName);

    ExecutableNetwork exeNetwork = ie.LoadNetwork(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetConfig("TARGET_FALLBACK"));
    std::string targets = p;
    auto expectedTargets = deviceName + "," + CommonTestUtils::DEVICE_CPU;

    std::cout << "Exe network fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}

//
// QueryNetwork with HETERO on particular device
//
bool supportsDeviceID(Core &ie, const std::string &deviceName) {
    auto supportedConfigKeys = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
    return supportedConfigKeys.end() != std::find(std::begin(supportedConfigKeys),
                                                  std::end(supportedConfigKeys),
                                                  CONFIG_KEY(DEVICE_ID));
}


TEST_P(IEClassQueryNetworkTest, QueryNetworkHETEROWithDeviceIDNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        ASSERT_NO_THROW(ie.QueryNetwork(actualNetwork, CommonTestUtils::DEVICE_HETERO,
                                        {{"TARGET_FALLBACK", deviceName + "." + deviceIDs[0] + "," + deviceName}}));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithDeviceID) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        try {
            ie.QueryNetwork(simpleNetwork, deviceName + ".0");
        } catch (const InferenceEngine::Exception & ex) {
            std::string message = ex.what();
            ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
        }
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithBigDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.QueryNetwork(actualNetwork, deviceName + ".110"), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithInvalidDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.QueryNetwork(actualNetwork, deviceName + ".l0"), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkHETEROWithBigDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.QueryNetwork(actualNetwork, CommonTestUtils::DEVICE_HETERO,
                                     {{"TARGET_FALLBACK", deviceName + ".100," + deviceName}}), Exception);
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with HETERO on particular device
//
TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROWithDeviceIDNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        std::string heteroDevice = CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName + "." + deviceIDs[0] + "," + deviceName;
        ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, heteroDevice));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkWithDeviceIDNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        ASSERT_NO_THROW(ie.LoadNetwork(simpleNetwork, deviceName + "." + deviceIDs[0]));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkWithBigDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.LoadNetwork(actualNetwork, deviceName + ".10"), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkWithInvalidDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.LoadNetwork(actualNetwork, deviceName + ".l0"), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROWithBigDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.LoadNetwork(actualNetwork, "HETERO",
                                    {{"TARGET_FALLBACK", deviceName + ".100," + CommonTestUtils::DEVICE_CPU}}), Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROAndDeviceIDThrows) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.LoadNetwork(actualNetwork, CommonTestUtils::DEVICE_HETERO,
                                    {{"TARGET_FALLBACK",     deviceName + "," + CommonTestUtils::DEVICE_CPU},
                                     {CONFIG_KEY(DEVICE_ID), "110"}}), Exception);
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with HETERO on MULTI combinations particular device
//

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROwithMULTINoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto &&device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        std::string targetFallback(CommonTestUtils::DEVICE_MULTI + std::string(",") + deviceName);
        ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, CommonTestUtils::DEVICE_HETERO, {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                {"TARGET_FALLBACK",                   targetFallback}}));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkMULTIwithHETERONoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto &&device : availableDevices) {
            devices += CommonTestUtils::DEVICE_HETERO + std::string(".") + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, CommonTestUtils::DEVICE_MULTI, {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                {"TARGET_FALLBACK",                   deviceName + "," + deviceName}}));
    } else {
        GTEST_SKIP();
    }
}

//
// QueryNetwork with HETERO on MULTI combinations particular device
//

TEST_P(IEClassLoadNetworkTest, QueryNetworkHETEROWithMULTINoThrow_V10) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto &&device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        auto function = multinputNetwork.getFunction();
        ASSERT_NE(nullptr, function);
        std::unordered_set<std::string> expectedLayers;
        for (auto &&node : function->get_ops()) {
            expectedLayers.emplace(node->get_friendly_name());
        }
        QueryNetworkResult result;
        std::string targetFallback(CommonTestUtils::DEVICE_MULTI + std::string(",") + deviceName);
        ASSERT_NO_THROW(result = ie.QueryNetwork(multinputNetwork, CommonTestUtils::DEVICE_HETERO, {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                {"TARGET_FALLBACK",                   targetFallback}}));

        std::unordered_set<std::string> actualLayers;
        for (auto &&layer : result.supportedLayersMap) {
            actualLayers.emplace(layer.first);
        }
        ASSERT_EQ(expectedLayers, actualLayers);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, QueryNetworkMULTIWithHETERONoThrow_V10) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto &&device : availableDevices) {
            devices += "HETERO." + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        auto function = multinputNetwork.getFunction();
        ASSERT_NE(nullptr, function);
        std::unordered_set<std::string> expectedLayers;
        for (auto &&node : function->get_ops()) {
            expectedLayers.emplace(node->get_friendly_name());
        }
        QueryNetworkResult result;
        ASSERT_NO_THROW(result = ie.QueryNetwork(multinputNetwork, CommonTestUtils::DEVICE_MULTI, {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                {"TARGET_FALLBACK",                   deviceName + "," + deviceName}}));

        std::unordered_set<std::string> actualLayers;
        for (auto &&layer : result.supportedLayersMap) {
            actualLayers.emplace(layer.first);
        }
        ASSERT_EQ(expectedLayers, actualLayers);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkAfterCoreRecreateTest, LoadAfterRecreateCoresAndPlugins) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    {
        auto versions = ie.GetVersions(std::string(CommonTestUtils::DEVICE_MULTI) + ":" + deviceName + "," + CommonTestUtils::DEVICE_CPU);
        ASSERT_EQ(3, versions.size());
    }
    std::map<std::string, std::string> config;
    if (deviceName == CommonTestUtils::DEVICE_CPU) {
        config.insert({"CPU_THREADS_NUM", "3"});
    }
    ASSERT_NO_THROW({
                        Core ie;
                        std::string name = actualNetwork.getInputsInfo().begin()->first;
                        actualNetwork.getInputsInfo().at(name)->setPrecision(Precision::U8);
                        auto executableNetwork = ie.LoadNetwork(actualNetwork, deviceName, config);
                    });
};
} // namespace BehaviorTestsDefinitions

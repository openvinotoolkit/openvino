// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <details/ie_cnn_network_tools.h>
#include <details/ie_cnn_network_iterator.hpp>
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <tests_common.hpp>
#include <memory>
#include <fstream>
#include <test_model_path.hpp>
#include <hetero/hetero_plugin_config.hpp>
#include <graph_tools.hpp>
#include <functional_test_utils/plugin_cache.hpp>
#include <multi-device/multi_device_config.hpp>

#include <ngraph/function.hpp>
#include <ngraph/op/subtract.hpp>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

#ifdef ENABLE_UNICODE_PATH_SUPPORT
#include <iostream>
#define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#include <codecvt>
#endif

using namespace testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace InferenceEngine::PluginConfigParams;

#define CHECK_MULTI() do { \
                          try { \
                              Core ie; \
                              ie.GetVersions("MULTI"); \
                          } catch (...) { \
                            GTEST_SKIP(); \
                          } \
                      } while(false)\

class IEClassBasicTest : public TestsCommon {
public:
    void SetUp() override {
        // To close loaded devices.
        PluginCache::get().reset();
    }
};

class IEClassBasicTestP : public IEClassBasicTest, public WithParamInterface<std::pair<std::string, std::string> > {
protected:
    std::string pluginName;
    std::string deviceName;
public:
    void SetUp() override {
        IEClassBasicTest::SetUp();

        pluginName = GetParam().first + IE_BUILD_POSTFIX;
        deviceName = GetParam().second;
    }
};

class IEClassNetworkTest : public IEClassBasicTest {
public:
    void SetUp() override {
        IEClassBasicTest::SetUp();

        // Generic network - GoogleNet V1
        {
            std::shared_ptr<ngraph::Function> fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
            ASSERT_NO_THROW(actualNetwork = CNNNetwork(fnPtr));
        }

        // Quite simple network
        {
            std::shared_ptr<ngraph::Function> fnPtr = ngraph::builder::subgraph::makeSingleConv();
            fnPtr->set_friendly_name("simpleNetwork");
            ASSERT_NO_THROW(simpleNetwork = CNNNetwork(fnPtr));
        }

        // miltiinput to substruct network
        {
            auto fnPtr = ngraph::builder::subgraph::make2InputSubtract();
            irv10Network = InferenceEngine::CNNNetwork(fnPtr);
        }
    }

    void setHeteroNetworkAffinity(const std::string& target) {
        InferenceEngine::InputsDataMap networkInputs = actualNetwork.getInputsInfo();

        CNNLayerPtr layer;
        for (auto input : networkInputs) {
            InputInfo::Ptr q = input.second;
            DataPtr p = q->getInputData();
            layer = p->getInputTo().begin()->second;
        }

        std::map<std::string, std::string> deviceMapping = {
            {"Convololution_4",   target},
            {"Convololution_7",   "CPU"},
            {"Relu_5",   "CPU"},
            {"Relu_8",   target},
            {"Concat_9", "CPU"}
        };

        CNNNetDFS(layer, [&](const CNNLayerPtr &layer) {
            auto it = deviceMapping.find(layer->name);
            if (it != deviceMapping.end()) {
                layer->affinity = it->second;
            } else {
                layer->affinity = "CPU";
            }
        });
    }

    CNNNetwork actualNetwork;
    CNNNetwork simpleNetwork;
    CNNNetwork irv10Network;

};

class IEClassNetworkTestP : public IEClassNetworkTest, public WithParamInterface<std::string> {
protected:
    std::string deviceName;
public:
    void SetUp() override {
        IEClassNetworkTest::SetUp();

        deviceName = GetParam();
    }
};

//
// Create and register plugins
//

TEST_F(IEClassBasicTest, smoke_createDefault) {
    ASSERT_NO_THROW(Core ie);
}

TEST_P(IEClassBasicTestP, registerExistingPluginThrows) {
    Core ie;
    ASSERT_THROW(ie.RegisterPlugin(pluginName, deviceName), InferenceEngineException);
}

TEST_P(IEClassBasicTestP, registerNewPluginNoThrows) {
    Core ie;
    ASSERT_NO_THROW(ie.RegisterPlugin(pluginName, "NEW_DEVICE_NAME"));
    ASSERT_NO_THROW(ie.GetMetric("NEW_DEVICE_NAME", METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
}

TEST_F(IEClassBasicTest, smoke_registerExistingPluginFileThrows) {
    Core ie;
    ASSERT_THROW(ie.RegisterPlugins("nonExistPlugins.xml"), InferenceEngineException);
}

TEST_F(IEClassBasicTest, smoke_createNonExistingConfigThrows) {
    ASSERT_THROW(Core ie("nonExistPlugins.xml"), InferenceEngineException);
}

#if defined __linux__  && !defined(__APPLE__)

TEST_F(IEClassBasicTest, smoke_createMockEngineConfigNoThrows) {
    ASSERT_NO_THROW(Core ie(TestDataHelpers::get_data_path() + "/ie_class/mock_engine_valid.xml"));
}

TEST_F(IEClassBasicTest, smoke_createMockEngineConfigThrows) {
    ASSERT_THROW(Core ie(TestDataHelpers::get_data_path() + "/ie_class/mock_engine.xml"), InferenceEngineException);
}

#endif

#ifdef ENABLE_UNICODE_PATH_SUPPORT

TEST_P(IEClassBasicTestP, smoke_registerPluginsXMLUnicodePath) {
// TODO: Issue: 31197 Remove this code
#if defined(_WIN32) || defined(_WIN64)
    if (deviceName == CommonTestUtils::DEVICE_MYRIAD) {
        GTEST_SKIP();
    }
#endif

    std::string pluginXML = TestDataHelpers::get_data_path() + "/ie_class/mock_engine_valid.xml";

    for (std::size_t testIndex = 0; testIndex < CommonTestUtils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix  = L"_" + CommonTestUtils::test_unicode_postfix_vector[testIndex];
        std::wstring pluginsXmlW = CommonTestUtils::addUnicodePostfixToPath(pluginXML, postfix);

        try {
            bool is_copy_successfully;
            is_copy_successfully = CommonTestUtils::copyFile(pluginXML, pluginsXmlW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << pluginXML << "' to '" << wStringtoMBCSstringChar(pluginsXmlW) << "'";
            }

            GTEST_COUT << "Test " << testIndex << std::endl;

            Core ie;
            GTEST_COUT << "Core created " << testIndex << std::endl;
            ASSERT_NO_THROW(ie.RegisterPlugins(wStringtoMBCSstringChar(pluginsXmlW)));
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
        catch (const InferenceEngine::details::InferenceEngineException &e_next) {
            CommonTestUtils::removeFile(pluginsXmlW);
            FAIL() << e_next.what();
        }
    }
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

//
// GetVersions()
//

TEST_P(IEClassBasicTestP, getVersionsByExactDeviceNoThrow) {
    Core ie;
    ASSERT_NO_THROW(ie.GetVersions(deviceName + ".0"));
}

TEST_P(IEClassBasicTestP, getVersionsByDeviceClassNoThrow) {
    Core ie;
    ASSERT_NO_THROW(ie.GetVersions(deviceName));
}

TEST_P(IEClassBasicTestP, getVersionsNonEmpty) {
    Core ie;
    ASSERT_EQ(2, ie.GetVersions("HETERO:" + deviceName).size());
}

//
// UnregisterPlugin
//

TEST_P(IEClassBasicTestP, unregisterExistingPluginNoThrow) {
    Core ie;
    // device instance is not created yet
    ASSERT_THROW(ie.UnregisterPlugin(deviceName), InferenceEngineException);

    // make the first call to IE which created device instance
    ie.GetVersions(deviceName);
    // now, we can unregister device
    ASSERT_NO_THROW(ie.UnregisterPlugin(deviceName));
}

TEST_P(IEClassBasicTestP, accessToUnregisteredPluginThrows) {
    Core ie;
    ASSERT_THROW(ie.UnregisterPlugin(deviceName), InferenceEngineException);
    ASSERT_NO_THROW(ie.GetVersions(deviceName));
    ASSERT_NO_THROW(ie.UnregisterPlugin(deviceName));
    ASSERT_NO_THROW(ie.SetConfig({ }, deviceName));
    ASSERT_NO_THROW(ie.GetVersions(deviceName));
    ASSERT_NO_THROW(ie.UnregisterPlugin(deviceName));
}

TEST_F(IEClassBasicTest, smoke_unregisterNonExistingPluginThrows) {
    Core ie;
    ASSERT_THROW(ie.UnregisterPlugin("unkown_device"), InferenceEngineException);
}

//
// SetConfig
//

TEST_P(IEClassBasicTestP, SetConfigAllThrows) {
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({ { "unsupported_key", "4" } }));
    ASSERT_ANY_THROW(ie.GetVersions(deviceName));
}

TEST_P(IEClassBasicTestP, SetConfigForUnRegisteredDeviceThrows) {
    Core ie;
    ASSERT_THROW(ie.SetConfig({ { "unsupported_key", "4" } }, "unregistered_device"), InferenceEngineException);
}

TEST_P(IEClassBasicTestP, SetConfigNoThrow) {
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({ { KEY_PERF_COUNT, YES } }, deviceName));
}

TEST_P(IEClassBasicTestP, SetConfigAllNoThrow) {
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({ { KEY_PERF_COUNT, YES } }));
    ASSERT_NO_THROW(ie.GetVersions(deviceName));
}

TEST_F(IEClassBasicTest, smoke_SetConfigHeteroThrows) {
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({ { KEY_PERF_COUNT, YES } }, "HETERO"));
}

TEST_P(IEClassBasicTestP, SetConfigHeteroTargetFallbackThrows) {
    Core ie;
    ASSERT_NO_THROW(ie.SetConfig({ { "TARGET_FALLBACK", deviceName } }, "HETERO"));
}

TEST_F(IEClassBasicTest, smoke_SetConfigHeteroNoThrow) {
    Core ie;
    bool value = false;

    ASSERT_NO_THROW(ie.SetConfig({ { HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), YES } }, "HETERO"));
    ASSERT_NO_THROW(value = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_TRUE(value);

    ASSERT_NO_THROW(ie.SetConfig({ { HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), NO } }, "HETERO"));
    ASSERT_NO_THROW(value = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_FALSE(value);
}

//
// ImportNetwork
//

TEST_P(IEClassBasicTestP, ImportNetworkThrows) {
    Core ie;

    if (deviceName == "CPU" || deviceName == "FPGA") {
        ASSERT_THROW(ie.ImportNetwork("model", deviceName), InferenceEngineException);
    }
}

TEST_F(IEClassBasicTest, smoke_ImportNetworkHeteroThrows) {
    Core ie;

    ASSERT_THROW(ie.ImportNetwork("model", "HETERO"), InferenceEngineException);
}

TEST_F(IEClassBasicTest, smoke_ImportNetworkMultiThrows) {
    CHECK_MULTI();
    InferenceEngine::Core ie;
    ASSERT_THROW(ie.ImportNetwork("model", "MULTI"), InferenceEngineException);
}

TEST_P(IEClassBasicTestP, ImportNetworkWithNullContextThrows) {
    Core ie;
    RemoteContext::Ptr context = nullptr;
    std::istringstream stream("None");
    ASSERT_THROW(ie.ImportNetwork(stream, context, {}), InferenceEngineException);
}

//
// LoadNetwork
//

TEST_P(IEClassNetworkTestP, LoadNetworkActualNoThrow) {
    Core ie;
    ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork,  deviceName));
}

TEST_P(IEClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    Core ie;
    ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, "HETERO:" + deviceName ));
}

TEST_P(IEClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    Core ie;
    ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, "HETERO", { { "TARGET_FALLBACK", deviceName } }));
}

#define SKIP_IF_NOT_IMPLEMENTED(...) do {                                       \
    try {                                                                       \
        __VA_ARGS__;                                                            \
    } catch(InferenceEngine::details::InferenceEngineException ieException) {   \
        auto notImplementedExceptionIsThrown =                                  \
            std::string::npos != std::string{ieException.what()}                \
            .find(std::string{"[NOT_IMPLEMENTED] "});                           \
        if (notImplementedExceptionIsThrown) {                                  \
            GTEST_SKIP();                                                       \
        } else {                                                                \
            FAIL() << "thrown from expression: " # __VA_ARGS__ << std::endl     \
            << "what: " << ieException.what();                                  \
        }                                                                       \
    }                                                                           \
} while(0)

//
// ImportExportNetwork
//

using IEClassImportExportTestP = IEClassNetworkTestP;

TEST_P(IEClassImportExportTestP, smoke_ImportNetworkNoThrowIfNoDeviceName) {
    Core ie;
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
    SKIP_IF_NOT_IMPLEMENTED(executableNetwork.Export(strm));
    if (!strm.str().empty() && deviceName.find("FPGA") != std::string::npos) {
        SKIP_IF_NOT_IMPLEMENTED(executableNetwork = ie.ImportNetwork(strm));
    }
    if (nullptr != static_cast<IExecutableNetwork::Ptr&>(executableNetwork)) {
        ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    }
}

TEST_P(IEClassImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    Core ie;
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
    SKIP_IF_NOT_IMPLEMENTED(executableNetwork.Export(strm));
    SKIP_IF_NOT_IMPLEMENTED(executableNetwork = ie.ImportNetwork(strm, deviceName));
    if (nullptr != static_cast<IExecutableNetwork::Ptr&>(executableNetwork)) {
        ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    }
}

TEST_P(IEClassImportExportTestP, smoke_ExportUsingFileNameImportFromStreamNoThrowWithDeviceName) {
    Core ie;
    ExecutableNetwork executableNetwork;
    std::string fileName{"ExportedNetwork"};
    {
        ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(simpleNetwork, deviceName));
        SKIP_IF_NOT_IMPLEMENTED(executableNetwork.Export(fileName));
    }
    if (CommonTestUtils::fileExists(fileName)) {
        {
            std::ifstream strm(fileName);
            SKIP_IF_NOT_IMPLEMENTED(executableNetwork = ie.ImportNetwork(strm, deviceName));
        }
        ASSERT_EQ(0, remove(fileName.c_str()));
    }
    if (nullptr != static_cast<IExecutableNetwork::Ptr&>(executableNetwork)) {
        ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    }
}

//
// QueryNetwork
//

TEST_P(IEClassNetworkTestP, QueryNetworkActualThrows) {
    Core ie;
    ASSERT_NO_THROW(ie.QueryNetwork(actualNetwork, "HETERO:" + deviceName));
}

TEST_P(IEClassNetworkTestP, QueryNetworkActualNoThrow) {
    Core ie;
    ASSERT_NO_THROW(ie.QueryNetwork(actualNetwork, deviceName));
}

TEST_P(IEClassNetworkTestP, QueryNetworkHeteroActualNoThrow) {
    Core ie;
    QueryNetworkResult res;
    ASSERT_NO_THROW(res = ie.QueryNetwork(actualNetwork, "HETERO", { { "TARGET_FALLBACK", deviceName } }));
    ASSERT_LT(0, res.supportedLayersMap.size());
}

TEST_P(IEClassNetworkTestP, QueryNetworkMultiThrows) {
    CHECK_MULTI();
    Core ie;
    ASSERT_THROW(ie.QueryNetwork(actualNetwork, "MULTI"), InferenceEngineException);
}

//
// IE Class GetMetric / GetConfig
//

class IEClassGetMetricTest : public TestsCommon, public WithParamInterface<std::string> {
protected:
    std::string deviceName;

public:
    void SetUp() override {
        // To close loaded devices.
        PluginCache::get().reset();

        deviceName = GetParam();
    }
};

#define ASSERT_METRIC_SUPPORTED(metricName)                              \
    {                                                                    \
        std::vector<std::string> metrics =                               \
            ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));     \
        auto it = std::find(metrics.begin(), metrics.end(), metricName); \
        ASSERT_NE(metrics.end(), it);                                    \
    }

TEST_F(IEClassBasicTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    Core ie;
    Parameter p;
    std::string deviceName = "HETERO";

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> t = p;

    std::cout << "Supported HETERO metrics: " << std::endl;
    for (auto && str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_F(IEClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroNoThrow) {
    Core ie;
    Parameter p;
    std::string deviceName = "HETERO";

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> t = p;

    std::cout << "Supported HETERO config keys: " << std::endl;
    for (auto && str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_F(IEClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroThrows) {
    Core ie;

    ASSERT_THROW(ie.GetMetric("HETERO:CPU", METRIC_KEY(SUPPORTED_CONFIG_KEYS)), InferenceEngineException);
}

using IEClassGetMetricTest_SUPPORTED_METRICS = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_SUPPORTED_METRICS, GetMetricAndPrintNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> t = p;

    std::cout << "Supported metrics: " << std::endl;
    for (auto && str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

using IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricAndPrintNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> t = p;

    std::cout << "Supported config values: " << std::endl;
    for (auto && str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

using IEClassGetMetricTest_AVAILABLE_DEVICES = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_AVAILABLE_DEVICES, GetMetricAndPrintNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)));
    std::vector<std::string> t = p;

    std::cout << "Available devices: " << std::endl;
    for (auto && str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(AVAILABLE_DEVICES));
}

using IEClassGetMetricTest_FULL_DEVICE_NAME = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_FULL_DEVICE_NAME, GetMetricAndPrintNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(FULL_DEVICE_NAME)));
    std::string t = p;
    std::cout << "Full device name: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(FULL_DEVICE_NAME));
}

using IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES, GetMetricAndPrintNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));
    std::vector<std::string> t = p;

    std::cout << "Optimization capabilities: " << std::endl;
    for (auto && str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
}

using IEClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS)));
    unsigned int t = p;

    std::cout << "Number of waiting infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS));
}

using IEClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS)));
    unsigned int t = p;

    std::cout << "Number of executing infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS));
}

using IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
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

using IEClassGetMetricTest_RANGE_FOR_STREAMS = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_RANGE_FOR_STREAMS, GetMetricAndPrintNoThrow) {
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

using IEClassGetMetricTest_ThrowUnsupported = IEClassGetMetricTest;
TEST_P(IEClassGetMetricTest_ThrowUnsupported,GetMetricThrow) {
    Core ie;
    Parameter p;

    ASSERT_THROW(p = ie.GetMetric(deviceName, "unsupported_metric"), InferenceEngineException);
}

using IEClassGetConfigTest = IEClassGetMetricTest;
TEST_P(IEClassGetConfigTest, GetConfigNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto && confKey : configValues) {
        Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.GetConfig(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

using IEClassGetConfigTest = IEClassGetMetricTest;
TEST_P(IEClassGetConfigTest, GetConfigHeteroNoThrow) {
    Core ie;
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto && confKey : configValues) {
        ASSERT_NO_THROW(ie.GetConfig(deviceName, confKey));
    }
}

using IEClassGetConfigTest_ThrowUnsupported = IEClassGetMetricTest;
TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigHeteroThrow) {
    Core ie;
    Parameter p;

    ASSERT_THROW(p = ie.GetConfig("HETERO", "unsupported_config"), InferenceEngineException);
}

using IEClassGetConfigTest_ThrowUnsupported = IEClassGetMetricTest;
TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigHeteroWithDeviceThrow) {
    Core ie;
    Parameter p;

    ASSERT_THROW(p = ie.GetConfig("HETERO:" + deviceName, HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)), InferenceEngineException);
}

using IEClassGetConfigTest_ThrowUnsupported = IEClassGetMetricTest;
TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigThrow) {
    Core ie;
    Parameter p;

    ASSERT_THROW(p = ie.GetConfig(deviceName, "unsupported_config"), InferenceEngineException);
}

using IEClassGetAvailableDevices = IEClassGetMetricTest;
TEST_P(IEClassGetAvailableDevices, GetAvailableDevicesNoThrow) {
    Core ie;
    std::vector<std::string> devices;

    ASSERT_NO_THROW(devices = ie.GetAvailableDevices());

    bool deviceFound = false;
    std::cout << "Available devices: " << std::endl;
    for (auto && device : devices) {
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

class IEClassExecutableNetworkGetMetricTest : public IEClassNetworkTest, public WithParamInterface<std::string> {
protected:
    std::string deviceName;

public:
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        deviceName = GetParam();
    }
};

class IEClassExecutableNetworkGetMetricTestForSpecificConfig : public IEClassNetworkTest,
public WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>> {
protected:
    std::string deviceName;
    std::string configKey;
    std::string configValue;
public:
    virtual void SetUp() {
        IEClassNetworkTest::SetUp();
        deviceName = get<0>(GetParam());
        configKey = get<1>(GetParam()).first;
        configValue = get<1>(GetParam()).second;
    }
};

#define ASSERT_EXEC_METRIC_SUPPORTED(metricName)                         \
    {                                                                    \
        std::vector<std::string> metrics =                               \
            exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS));         \
        auto it = std::find(metrics.begin(), metrics.end(), metricName); \
        ASSERT_NE(metrics.end(), it);                                    \
    }

using IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassExecutableNetworkGetMetricTest;
TEST_P(IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    std::cout << "Supported config keys: " << std::endl;
    for (auto && conf : configValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, configValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

using IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = IEClassExecutableNetworkGetMetricTest;
TEST_P(IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,GetMetricNoThrow) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> metricValues = p;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto && conf : metricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, metricValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

using IEClassExecutableNetworkGetMetricTest_NETWORK_NAME = IEClassExecutableNetworkGetMetricTest;
TEST_P(IEClassExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
    ASSERT_EQ("simpleNetwork", networkname);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));
}

using IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = IEClassExecutableNetworkGetMetricTest;
TEST_P(IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)));
    unsigned int value = p;

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
}

using IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported = IEClassExecutableNetworkGetMetricTest;
TEST_P(IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.GetMetric("unsupported_metric"), InferenceEngineException);
}

using IEClassExecutableNetworkGetConfigTest = IEClassExecutableNetworkGetMetricTest;
TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto && confKey : configValues) {
        Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.GetConfig(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigThrows) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.GetConfig("unsupported_config"), InferenceEngineException);
}

using IEClassExecutableNetworkSetConfigTest = IEClassExecutableNetworkGetMetricTest;
TEST_P(IEClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.SetConfig({ { "unsupported_config", "some_value" } }), InferenceEngineException);
}

using IEClassExecutableNetworkSupportedConfigTest = IEClassExecutableNetworkGetMetricTestForSpecificConfig;
TEST_P(IEClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_NO_THROW(exeNetwork.SetConfig({ { configKey, configValue } }));
    ASSERT_NO_THROW(p = exeNetwork.GetConfig( configKey ));
    ASSERT_EQ(p, configValue);
}

using IEClassExecutableNetworkUnsupportedConfigTest = IEClassExecutableNetworkGetMetricTestForSpecificConfig;
TEST_P(IEClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    Core ie;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.SetConfig({ { configKey, configValue } }), InferenceEngineException);
}

using IEClassExecutableNetworkGetConfigTest = IEClassExecutableNetworkGetMetricTest;
TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
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

//
// Hetero Executable network case
//

class IEClassHeteroExecutableNetworkGetMetricTest : public IEClassNetworkTest, public WithParamInterface<std::string> {
protected:
    std::string deviceName;
    std::string heteroDeviceName;

public:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        IEClassNetworkTest::SetUp();
        deviceName = GetParam();
        heteroDeviceName = "HETERO:" + deviceName + ",CPU";
    }
};

using IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassHeteroExecutableNetworkGetMetricTest;
TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    Core ie;
    Parameter pHetero, pDevice;

    ExecutableNetwork heteroExeNetwork = ie.LoadNetwork(actualNetwork, heteroDeviceName);
    ExecutableNetwork deviceExeNetwork = ie.LoadNetwork(actualNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> heteroConfigValues = pHetero, deviceConfigValues = pDevice;

    std::cout << "Supported config keys: " << std::endl;
    for (auto && conf : heteroConfigValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, heteroConfigValues.size());

    // check that all device config values are present in hetero case
    for (auto && deviceConf : deviceConfigValues) {
        auto it = std::find(heteroConfigValues.begin(), heteroConfigValues.end(), deviceConf);
        ASSERT_TRUE(it != heteroConfigValues.end());

        Parameter heteroConfigValue = heteroExeNetwork.GetConfig(deviceConf);
        Parameter deviceConfigValue = deviceExeNetwork.GetConfig(deviceConf);

        // HETERO returns EXCLUSIVE_ASYNC_REQUESTS as a boolean value
        if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) != deviceConf)
            ASSERT_EQ(deviceConfigValue, heteroConfigValue);
    }
}

using IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS = IEClassHeteroExecutableNetworkGetMetricTest;
TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    Core ie;
    Parameter pHetero, pDevice;

    ExecutableNetwork heteroExeNetwork = ie.LoadNetwork(actualNetwork, heteroDeviceName);
    ExecutableNetwork deviceExeNetwork = ie.LoadNetwork(actualNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> heteroMetricValues = pHetero, deviceMetricValues = pDevice;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto && conf : heteroMetricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, heteroMetricValues.size());

    const std::vector<std::string> heteroSpecificMetrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS)
    };

    // check that all device metric values are present in hetero case
    for (auto && deviceMetricName : deviceMetricValues) {
        auto it = std::find(heteroMetricValues.begin(), heteroMetricValues.end(), deviceMetricName);
        ASSERT_TRUE(it != heteroMetricValues.end());

        Parameter heteroMetricValue = heteroExeNetwork.GetMetric(deviceMetricName);
        Parameter deviceMetricValue = deviceExeNetwork.GetMetric(deviceMetricName);

        if (std::find(heteroSpecificMetrics.begin(), heteroSpecificMetrics.end(), deviceMetricName) ==
            heteroSpecificMetrics.end())
            ASSERT_TRUE(heteroMetricValue == deviceMetricValue);
    }
}

using IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME = IEClassHeteroExecutableNetworkGetMetricTest;
TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    Core ie;
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
}

using IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK = IEClassHeteroExecutableNetworkGetMetricTest;
TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    Core ie;
    Parameter p;

    setHeteroNetworkAffinity(deviceName);

    ExecutableNetwork exeNetwork = ie.LoadNetwork(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetConfig("TARGET_FALLBACK"));
    std::string targets = p;
    auto expectedTargets = deviceName + ",CPU";

    std::cout << "Exe network fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}

//
// QueryNetwork with HETERO on particular device
//

namespace {

bool supportsDeviceID(Core & ie, const std::string & deviceName) {
    auto supportedConfigKeys = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
    return supportedConfigKeys.end() != std::find(std::begin(supportedConfigKeys),
                                                  std::end(supportedConfigKeys),
                                                  CONFIG_KEY(DEVICE_ID));
}

bool supportsAvaliableDevices(Core & ie, const std::string & deviceName) {
    auto supportedMetricKeys = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
    return supportedMetricKeys.end() != std::find(std::begin(supportedMetricKeys),
                                                  std::end(supportedMetricKeys),
                                                  METRIC_KEY(AVAILABLE_DEVICES));
}

}

class IEClassQueryNetworkTest : public IEClassNetworkTest, public WithParamInterface<std::string> {
protected:
    std::string deviceName;
public:
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        deviceName = GetParam();
    }
};

TEST_P(IEClassQueryNetworkTest, QueryNetworkHETEROWithDeviceIDNoThrow) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        ASSERT_NO_THROW(ie.QueryNetwork(actualNetwork, "HETERO",
            { { "TARGET_FALLBACK", deviceName + "." + deviceIDs[0] + ",CPU" }}));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithDeviceID) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_NO_THROW(ie.QueryNetwork(simpleNetwork, deviceName + ".0"));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithBigDeviceIDThrows) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.QueryNetwork(actualNetwork, deviceName + ".110"), InferenceEngineException);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithInvalidDeviceIDThrows) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.QueryNetwork(actualNetwork, deviceName + ".l0"), InferenceEngineException);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkHETEROWithBigDeviceIDThrows) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.QueryNetwork(actualNetwork, "HETERO",
                                     { { "TARGET_FALLBACK", deviceName + ".100,CPU" }}), InferenceEngineException);
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with HETERO on particular device
//

using IEClassLoadNetworkTest = IEClassQueryNetworkTest;

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROWithDeviceIDNoThrow) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        std::string heteroDevice = "HETERO:" + deviceName + "." + deviceIDs[0] + ",CPU";
        ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, heteroDevice));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkWithDeviceIDNoThrow) {
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
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.LoadNetwork(actualNetwork, deviceName + ".10"), InferenceEngineException);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkWithInvalidDeviceIDThrows) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.LoadNetwork(actualNetwork, deviceName + ".l0"), InferenceEngineException);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROWithBigDeviceIDThrows) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.LoadNetwork(actualNetwork, "HETERO",
                                     { { "TARGET_FALLBACK", deviceName + ".100,CPU" } }), InferenceEngineException);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROAndDeviceIDThrows) {
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.LoadNetwork(actualNetwork, "HETERO",
                                     { { "TARGET_FALLBACK", deviceName + ",CPU" }, {CONFIG_KEY(DEVICE_ID), "110"}}), InferenceEngineException);
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with HETERO on MULTI combinations particular device
//

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROwithMULTINoThrow) {
    CHECK_MULTI();

    Core ie;
    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, "HETERO", {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                { "TARGET_FALLBACK", "MULTI,CPU" }}));
    } else {
        GTEST_SKIP();
    }

}

TEST_P(IEClassLoadNetworkTest, LoadNetworkMULTIwithHETERONoThrow) {
    CHECK_MULTI();
    Core ie;

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += "HETERO." + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        ASSERT_NO_THROW(ie.LoadNetwork(actualNetwork, "MULTI", {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                { "TARGET_FALLBACK", deviceName + ",CPU" }}));
    } else {
        GTEST_SKIP();
    }
}

//
// QueryNetwork with HETERO on MULTI combinations particular device
//

TEST_P(IEClassLoadNetworkTest, QueryNetworkHETEROwithMULTINoThrowv7) {
    CHECK_MULTI();
    Core ie;

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }

        QueryNetworkResult result;
        ASSERT_NO_THROW(result = ie.QueryNetwork(actualNetwork, "HETERO", {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                { "TARGET_FALLBACK", "MULTI,CPU" }}));

        for (auto && layer : result.supportedLayersMap) {
            IE_SUPPRESS_DEPRECATED_START
            EXPECT_NO_THROW(actualNetwork.getLayerByName(layer.first.c_str()));
            IE_SUPPRESS_DEPRECATED_END
        }
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, QueryNetworkMULTIwithHETERONoThrowv7) {
    CHECK_MULTI();
    Core ie;

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += "HETERO." + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }

        QueryNetworkResult result;
        ASSERT_NO_THROW(result = ie.QueryNetwork(actualNetwork, "MULTI", {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                { "TARGET_FALLBACK", deviceName + ",CPU" }}));

        for (auto && layer : result.supportedLayersMap) {
            IE_SUPPRESS_DEPRECATED_START
            EXPECT_NO_THROW(actualNetwork.getLayerByName(layer.first.c_str()));
            IE_SUPPRESS_DEPRECATED_END
        }
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, QueryNetworkHETEROwithMULTINoThrowv10) {
    CHECK_MULTI();
    Core ie;

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }

        QueryNetworkResult result;
        ASSERT_NO_THROW(result = ie.QueryNetwork(irv10Network, "HETERO", {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                { "TARGET_FALLBACK", "MULTI,CPU" }}));

        for (auto && layer : result.supportedLayersMap) {
            IE_SUPPRESS_DEPRECATED_START
            EXPECT_NO_THROW(irv10Network.getLayerByName(layer.first.c_str()));
            IE_SUPPRESS_DEPRECATED_END
        }
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, DISABLED_QueryNetworkMULTIwithHETERONoThrowv10) {
    CHECK_MULTI();
    Core ie;

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        for (auto&& device : availableDevices) {
            devices += "HETERO." + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }

        // TODO: remove once HETERO and MULTI support v10
        irv10Network.getLayerByName("param0");

        std::vector<std::string> names;
        if (auto ngraphFunction = irv10Network.getFunction()) {
            for (auto && op : irv10Network.getFunction()->get_ops()) {
                names.push_back(op->get_friendly_name());
            }
        } else {
            IE_SUPPRESS_DEPRECATED_START
            auto & inetwork = (ICNNNetwork&)irv10Network;
            details::CNNNetworkIterator i(&inetwork), end;
            while (i != end) {
                CNNLayerPtr layer = *i;
                names.push_back(layer->name);
                ++i;
            }
            IE_SUPPRESS_DEPRECATED_END
        }

        QueryNetworkResult result;
        ASSERT_NO_THROW(result = ie.QueryNetwork(irv10Network, "MULTI", {
                {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
                { "TARGET_FALLBACK", deviceName + ",CPU" }}));

        // check that all supported layers are in network
        for (auto && layer : result.supportedLayersMap) {
            EXPECT_NE(std::end(names), std::find(names.begin(), names.end(), layer.first));
        }

        // check that network layers are supported
        for (auto && name : names) {
            bool layerIsFound = result.supportedLayersMap.end() !=
                std::find_if(result.supportedLayersMap.begin(), result.supportedLayersMap.end(),
                    [&](const std::pair<std::string, std::string> & p) {
                        return name == p.first;
                    });
            EXPECT_TRUE(layerIsFound);
        }
    } else {
        GTEST_SKIP();
    }
}

using IEClassLoadNetworkAfterCoreRecreateTest = IEClassLoadNetworkTest;

TEST_P(IEClassLoadNetworkAfterCoreRecreateTest, LoadAfterRecreateCoresAndPlugins) {
    CHECK_MULTI();
    Core ie;
    {
        auto versions = ie.GetVersions("MULTI:" + deviceName + ",CPU");
        ASSERT_EQ(3, versions.size());
    }
    std::map<std::string, std::string> config;
    if (deviceName == CommonTestUtils::DEVICE_CPU) {
        config.insert({"CPU_THREADS_NUM", "3"});
    };
    ASSERT_NO_THROW({
        Core ie;
        std::string name = actualNetwork.getInputsInfo().begin()->first;
        actualNetwork.getInputsInfo().at(name)->setPrecision(Precision::U8);
        auto executableNetwork = ie.LoadNetwork(actualNetwork, deviceName, config);
    });
}

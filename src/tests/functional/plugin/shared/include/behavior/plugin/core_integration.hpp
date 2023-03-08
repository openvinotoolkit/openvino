// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <thread>

#include "base/behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/util/file_util.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#include <iostream>
#define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#include <codecvt>
#endif

namespace BehaviorTestsDefinitions {

#define ASSERT_METRIC_SUPPORTED_IE(metricName)                       \
{                                                                    \
    auto metrics =                               \
        ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();  \
    auto it = std::find(metrics.begin(), metrics.end(), metricName); \
    ASSERT_NE(metrics.end(), it);                                    \
}

inline bool supportsAvaliableDevices(InferenceEngine::Core& ie, const std::string& target_device) {
    auto supportedMetricKeys =
        ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
    return supportedMetricKeys.end() !=
           std::find(std::begin(supportedMetricKeys), std::end(supportedMetricKeys), METRIC_KEY(AVAILABLE_DEVICES));
}

inline bool supportsDeviceID(InferenceEngine::Core &ie, const std::string &target_device) {
    auto supportedConfigKeys = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
    return supportedConfigKeys.end() != std::find(std::begin(supportedConfigKeys),
                                                  std::end(supportedConfigKeys),
                                                  CONFIG_KEY(DEVICE_ID));
}

class IEClassBasicTestP : public BehaviorTestsUtils::IEPluginTestBase,
                          public ::testing::WithParamInterface<std::pair<std::string, std::string> > {
protected:
    std::string deviceName;
    std::string pluginName;

public:
    void SetUp() override {
        std::tie(pluginName, target_device) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        ov::test::behavior::APIBaseTest::SetUp();
        pluginName += IE_BUILD_POSTFIX;
        if (pluginName == (std::string("openvino_template_plugin") + IE_BUILD_POSTFIX)) {
            pluginName = ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(), pluginName);
        }
    }
};

class IEClassSetDefaultDeviceIDTest : public BehaviorTestsUtils::IEPluginTestBase,
                                      public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceID;

public:
    void SetUp() override {
        std::tie(target_device, deviceID) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
    }
};

using IEClassNetworkTestP = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassLoadNetworkTestWithThrow = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassQueryNetworkTest = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_SUPPORTED_METRICS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_AVAILABLE_DEVICES = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_FULL_DEVICE_NAME = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_DEVICE_GOPS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_DEVICE_TYPE = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_ThrowUnsupported = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetConfigTest = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetConfigTest_ThrowUnsupported = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetAvailableDevices = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassGetMetricTest_RANGE_FOR_STREAMS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassSetGlobalConfigTest = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassSpecificDeviceTestSetConfig = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassSpecificDeviceTestGetConfig = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassLoadNetworkAfterCoreRecreateTest = BehaviorTestsUtils::IEClassBaseTestP;

class IEClassSeveralDevicesTest : public BehaviorTestsUtils::IEPluginTestBase,
                                  public BehaviorTestsUtils::IEClassNetworkTest,
                                  public ::testing::WithParamInterface<std::vector<std::string>> {
public:
    std::vector<std::string> target_devices;
    void SetUp() override {
        target_device = CommonTestUtils::DEVICE_MULTI;
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        ov::test::behavior::APIBaseTest::SetUp();
        IEClassNetworkTest::SetUp();
        target_devices = GetParam();
    }
};

using IEClassSeveralDevicesTestLoadNetwork = IEClassSeveralDevicesTest;
using IEClassSeveralDevicesTestQueryNetwork = IEClassSeveralDevicesTest;
using IEClassSeveralDevicesTestDefaultCore = IEClassSeveralDevicesTest;

TEST(IEClassBasicTest, smoke_createDefault) {
    ASSERT_NO_THROW(InferenceEngine::Core  ie);
}

// TODO: CVS-68982
#ifndef OPENVINO_STATIC_LIBRARY

TEST_P(IEClassBasicTestP, registerExistingPluginThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_THROW(ie.RegisterPlugin(pluginName, target_device), InferenceEngine::Exception);
}

TEST_P(IEClassBasicTestP, registerNewPluginNoThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.RegisterPlugin(pluginName, "NEW_DEVICE_NAME"));
    ASSERT_NO_THROW(ie.GetMetric("NEW_DEVICE_NAME", METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
}

TEST(IEClassBasicTest, smoke_registerNonExistingPluginFileThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_THROW(ie.RegisterPlugins("nonExistPlugins.xml"), InferenceEngine::Exception);
}

TEST(IEClassBasicTest, smoke_createNonExistingConfigThrows) {
    ASSERT_THROW(InferenceEngine::Core  ie("nonExistPlugins.xml"), InferenceEngine::Exception);
}

inline std::string getPluginFile() {
    std::string filePostfix{"mock_engine_valid.xml"};
    std::string filename = CommonTestUtils::generateTestFilePrefix() + "_" + filePostfix;
    std::ostringstream stream;
    stream << "<ie><plugins><plugin name=\"mock\" location=\"";
    stream << ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
        std::string("mock_engine") + IE_BUILD_POSTFIX);
    stream << "\"></plugin></plugins></ie>";
    CommonTestUtils::createFile(filename, stream.str());
    return filename;
}

TEST(IEClassBasicTest, smoke_createMockEngineConfigNoThrows) {
    const std::string filename = getPluginFile();
    ASSERT_NO_THROW(InferenceEngine::Core  ie(filename));
    CommonTestUtils::removeFile(filename.c_str());
}

TEST(IEClassBasicTest, smoke_createMockEngineConfigThrows) {
    std::string filename = CommonTestUtils::generateTestFilePrefix() + "_mock_engine.xml";
    std::string content{"<ie><plugins><plugin location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_THROW(InferenceEngine::Core  ie(filename), InferenceEngine::Exception);
    CommonTestUtils::removeFile(filename.c_str());
}
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

TEST_P(IEClassBasicTestP, smoke_registerPluginsXMLUnicodePath) {
    const std::string pluginXML = getPluginFile();

    for (std::size_t testIndex = 0; testIndex < CommonTestUtils::test_unicode_postfix_vector.size(); testIndex++) {
        GTEST_COUT << testIndex;
        std::wstring postfix  = L"_" + CommonTestUtils::test_unicode_postfix_vector[testIndex];
        std::wstring pluginsXmlW = CommonTestUtils::addUnicodePostfixToPath(pluginXML, postfix);

        try {
            bool is_copy_successfully;
            is_copy_successfully = CommonTestUtils::copyFile(pluginXML, pluginsXmlW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << pluginXML << "' to '" << ov::util::wstring_to_string(pluginsXmlW) << "'";
            }

            GTEST_COUT << "Test " << testIndex << std::endl;

            InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
            GTEST_COUT << "Core created " << testIndex << std::endl;
            ASSERT_NO_THROW(ie.RegisterPlugins(ov::util::wstring_to_string(pluginsXmlW)));
            CommonTestUtils::removeFile(pluginsXmlW);
            ASSERT_NO_THROW(ie.GetVersions("mock")); // from pluginXM
            ASSERT_NO_THROW(ie.GetVersions(target_device));
            GTEST_COUT << "Plugin created " << testIndex << std::endl;

            ASSERT_NO_THROW(ie.RegisterPlugin(pluginName, "TEST_DEVICE"));
            ASSERT_NO_THROW(ie.GetVersions("TEST_DEVICE"));
            GTEST_COUT << "Plugin registered and created " << testIndex << std::endl;

            GTEST_COUT << "OK" << std::endl;
        }
        catch (const InferenceEngine::Exception&e_next) {
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

TEST_P(IEClassBasicTestP, getVersionsByExactDeviceNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.GetVersions(target_device + ".0"));
}

TEST_P(IEClassBasicTestP, getVersionsByDeviceClassNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.GetVersions(target_device));
}

TEST_P(IEClassBasicTestP, getVersionsNonEmpty) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_EQ(2, ie.GetVersions(CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device).size());
}

//
// UnregisterPlugin
//

TEST_P(IEClassBasicTestP, unregisterExistingPluginNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    // device instance is not created yet
    ASSERT_THROW(ie.UnregisterPlugin(target_device), InferenceEngine::Exception);

    // make the first call to IE which created device instance
    ie.GetVersions(target_device);
    // now, we can unregister device
    ASSERT_NO_THROW(ie.UnregisterPlugin(target_device));
}

TEST_P(IEClassBasicTestP, accessToUnregisteredPluginThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_THROW(ie.UnregisterPlugin(target_device), InferenceEngine::Exception);
    ASSERT_NO_THROW(ie.GetVersions(target_device));
    ASSERT_NO_THROW(ie.UnregisterPlugin(target_device));
    ASSERT_NO_THROW(ie.SetConfig({}, target_device));
    ASSERT_NO_THROW(ie.GetVersions(target_device));
    ASSERT_NO_THROW(ie.UnregisterPlugin(target_device));
}

TEST(IEClassBasicTest, smoke_unregisterNonExistingPluginThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_THROW(ie.UnregisterPlugin("unkown_device"), InferenceEngine::Exception);
}

//
// SetConfig
//

TEST_P(IEClassBasicTestP, SetConfigAllThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.SetConfig({{"unsupported_key", "4"}}));
    ASSERT_ANY_THROW(ie.GetVersions(target_device));
}

TEST_P(IEClassBasicTestP, SetConfigForUnRegisteredDeviceThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_THROW(ie.SetConfig({{"unsupported_key", "4"}}, "unregistered_device"), InferenceEngine::Exception);
}

TEST_P(IEClassBasicTestP, SetConfigNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}},
                                 target_device));
}

TEST_P(IEClassBasicTestP, SetConfigAllNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}}));
    ASSERT_NO_THROW(ie.GetVersions(target_device));
}

TEST(IEClassBasicTest, smoke_SetConfigHeteroThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}},
                                 CommonTestUtils::DEVICE_HETERO));
}

TEST_P(IEClassBasicTestP, SetGetConfigForTbbTerminateThrows) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    bool value = false;
    ASSERT_NO_THROW(ie.SetConfig({{CONFIG_KEY(FORCE_TBB_TERMINATE), CONFIG_VALUE(YES)}}));
    ASSERT_NO_THROW(value = ie.GetConfig(target_device, CONFIG_KEY(FORCE_TBB_TERMINATE)).as<bool>());
    ASSERT_TRUE(value);

    ASSERT_NO_THROW(ie.SetConfig({{CONFIG_KEY(FORCE_TBB_TERMINATE), CONFIG_VALUE(NO)}}));
    ASSERT_NO_THROW(value = ie.GetConfig(target_device, CONFIG_KEY(FORCE_TBB_TERMINATE)).as<bool>());
    ASSERT_FALSE(value);
}

TEST_P(IEClassBasicTestP, SetConfigHeteroTargetFallbackThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.SetConfig({{"TARGET_FALLBACK", target_device}}, CommonTestUtils::DEVICE_HETERO));
}

TEST(IEClassBasicTest, smoke_SetConfigHeteroNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    bool value = false;

    ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), InferenceEngine::PluginConfigParams::YES}},
                                 CommonTestUtils::DEVICE_HETERO));
    ASSERT_NO_THROW(value = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_TRUE(value);

    ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), InferenceEngine::PluginConfigParams::NO}},
                                 CommonTestUtils::DEVICE_HETERO));
    ASSERT_NO_THROW(value = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>());
    ASSERT_FALSE(value);
}

TEST_P(IEClassSpecificDeviceTestSetConfig, SetConfigSpecificDeviceNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    std::string deviceID, cleartarget_device;
    auto pos = target_device.find('.');
    if (pos != std::string::npos) {
        cleartarget_device = target_device.substr(0, pos);
        deviceID =  target_device.substr(pos + 1,  target_device.size());
    }
    if (!supportsDeviceID(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    auto deviceIDs = ie.GetMetric(cleartarget_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_SKIP();
    }

    ASSERT_NO_THROW(ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}}, target_device));
    std::string value;
    ASSERT_NO_THROW(value = ie.GetConfig(target_device, InferenceEngine::PluginConfigParams::KEY_PERF_COUNT).as<std::string>());
    ASSERT_EQ(value, InferenceEngine::PluginConfigParams::YES);
}

//
// ImportNetwork
//

TEST(IEClassBasicTest, smoke_ImportNetworkHeteroThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    ASSERT_THROW(ie.ImportNetwork("model", CommonTestUtils::DEVICE_HETERO), InferenceEngine::NetworkNotRead);
}

TEST(IEClassBasicTest, smoke_ImportNetworkMultiThrows) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_THROW(ie.ImportNetwork("model", CommonTestUtils::DEVICE_MULTI), InferenceEngine::NetworkNotRead);
}

TEST_P(IEClassBasicTestP, ImportNetworkWithNullContextThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::RemoteContext::Ptr context = nullptr;
    std::istringstream stream("None");
    ASSERT_THROW(ie.ImportNetwork(stream, context, {}), InferenceEngine::Exception);
}

//
// QueryNetwork
//

TEST_P(IEClassNetworkTestP, QueryNetworkActualThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.QueryNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device));
}

TEST_P(IEClassNetworkTestP, QueryNetworkActualNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    try {
        ie.QueryNetwork(actualCnnNetwork, target_device);
    } catch (const InferenceEngine::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(IEClassNetworkTestP, QueryNetworkWithKSO) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    try {
        auto rres = ie.QueryNetwork(ksoCnnNetwork, target_device);
        auto rl_map = rres.supportedLayersMap;
        auto func = ksoCnnNetwork.getFunction();
        for (const auto & op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
            }
        }
    } catch (const InferenceEngine::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(IEClassSeveralDevicesTestQueryNetwork, QueryNetworkActualSeveralDevicesNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    std::string cleartarget_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        cleartarget_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    auto deviceIDs = ie.GetMetric(cleartarget_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect DeviceID number" << std::endl;

    std::string multitarget_device = CommonTestUtils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : target_devices) {
        multitarget_device += dev_name;
        if (&dev_name != &(target_devices.back())) {
            multitarget_device += ",";
        }
    }
    ASSERT_NO_THROW(ie.QueryNetwork(actualCnnNetwork, multitarget_device));
}

TEST_P(IEClassNetworkTestP, SetAffinityWithConstantBranches) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

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
        InferenceEngine::CNNNetwork net(func);

        auto rres = ie.QueryNetwork(net,  target_device);
        auto rl_map = rres.supportedLayersMap;
        for (const auto & op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " <<  target_device;
            }
        }
        for (const auto & op : net.getFunction()->get_ops()) {
            std::string affinity = rl_map[op->get_friendly_name()];
            op->get_rt_info()["affinity"] = affinity;
        }
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(ksoCnnNetwork,  target_device);
    } catch (const InferenceEngine::NotImplemented & ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(IEClassNetworkTestP, SetAffinityWithKSO) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    try {
        auto rres = ie.QueryNetwork(ksoCnnNetwork,  target_device);
        auto rl_map = rres.supportedLayersMap;
        auto func = ksoCnnNetwork.getFunction();
        for (const auto & op : func->get_ops()) {
            if (!rl_map.count(op->get_friendly_name())) {
                FAIL() << "Op " << op->get_friendly_name() << " is not supported by " <<  target_device;
            }
        }
        for (const auto & op : ksoCnnNetwork.getFunction()->get_ops()) {
            std::string affinity = rl_map[op->get_friendly_name()];
            op->get_rt_info()["affinity"] = affinity;
        }
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(ksoCnnNetwork,  target_device);
    } catch (const InferenceEngine::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(IEClassNetworkTestP, QueryNetworkHeteroActualNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::QueryNetworkResult res;
    ASSERT_NO_THROW(res = ie.QueryNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK",  target_device}}));
    ASSERT_LT(0, res.supportedLayersMap.size());
}

TEST_P(IEClassNetworkTestP, DISABLED_QueryNetworkMultiThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    try {
        ie.QueryNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_MULTI);
    } catch (InferenceEngine::Exception& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("KEY_MULTI_DEVICE_PRIORITIES key is not set for"),
                            error.what());
    } catch (...) {
        FAIL() << "QueryNetwork failed for unexpected reson.";
    }
}

TEST(IEClassBasicTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;
    std::string  target_device = CommonTestUtils::DEVICE_HETERO;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_METRICS)));
    auto t = p.as<std::vector<std::string>>();

    std::cout << "Supported HETERO metrics: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(SUPPORTED_METRICS));
}

TEST(IEClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;
    std::string  target_device = CommonTestUtils::DEVICE_HETERO;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto t = p.as<std::vector<std::string>>();

    std::cout << "Supported HETERO config keys: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST(IEClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    // TODO: check
    std::string targetDevice = CommonTestUtils::DEVICE_HETERO + std::string(":") + CommonTestUtils::DEVICE_CPU;
    ASSERT_THROW(ie.GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)), InferenceEngine::Exception);
}

TEST_P(IEClassGetMetricTest_SUPPORTED_METRICS, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_METRICS)));
    auto t = p.as<std::vector<std::string>>();

    std::cout << "Supported metrics: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto t = p.as<std::vector<std::string>>();

    std::cout << "Supported config values: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(IEClassGetMetricTest_AVAILABLE_DEVICES, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)));
    auto t = p.as<std::vector<std::string>>();

    std::cout << "Available devices: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(AVAILABLE_DEVICES));
}

TEST_P(IEClassGetMetricTest_FULL_DEVICE_NAME, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(FULL_DEVICE_NAME)));
    auto t = p.as<std::string>();
    std::cout << "Full device name: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(FULL_DEVICE_NAME));
}

TEST_P(IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));
    auto t = p.as<std::vector<std::string>>();

    std::cout << "Optimization capabilities: " << std::endl;
    for (auto &&str : t) {
        std::cout << str << std::endl;
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
}

TEST_P(IEClassGetMetricTest_DEVICE_GOPS, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(DEVICE_GOPS)));
    auto t = p.as<std::map<InferenceEngine::Precision, float>>();

    std::cout << "Device GOPS: " << std::endl;
    for (auto &&kv : t) {
        std::cout << kv.first << ": " << kv.second << std::endl;
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(DEVICE_GOPS));
}

TEST_P(IEClassGetMetricTest_DEVICE_TYPE, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(DEVICE_TYPE)));
    auto t = p.as<InferenceEngine::Metrics::DeviceType>();

    std::cout << "Device Type: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(DEVICE_TYPE));
}

TEST_P(IEClassGetMetricTest_NUMBER_OF_WAITING_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS)));
    auto t = p.as<unsigned int>();

    std::cout << "Number of waiting infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS));
}

TEST_P(IEClassGetMetricTest_NUMBER_OF_EXEC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS)));
    auto t = p.as<unsigned int>();

    std::cout << "Number of executing infer requests: " << std::endl << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS));
}

TEST_P(IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)));
    auto t = p.as<std::tuple<unsigned int, unsigned int, unsigned int>>();

    unsigned int start = std::get<0>(t);
    unsigned int end = std::get<1>(t);
    unsigned int step = std::get<2>(t);

    std::cout << "Range for async infer requests: " << std::endl;
    std::cout << start << std::endl;
    std::cout << end << std::endl;
    std::cout << step << std::endl;
    std::cout << std::endl;

    ASSERT_LE(start, end);
    ASSERT_GE(step, 1u);
    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
}

TEST_P(IEClassGetMetricTest_RANGE_FOR_STREAMS, GetMetricAndPrintNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(RANGE_FOR_STREAMS)));
    auto t = p.as<std::tuple<unsigned int, unsigned int>>();

    unsigned int start = std::get<0>(t);
    unsigned int end = std::get<1>(t);

    std::cout << "Range for streams: " << std::endl;
    std::cout << start << std::endl;
    std::cout << end << std::endl;
    std::cout << std::endl;

    ASSERT_LE(start, end);
    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(RANGE_FOR_STREAMS));
}

TEST_P(IEClassGetMetricTest_ThrowUnsupported, GetMetricThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_THROW(p = ie.GetMetric(target_device, "unsupported_metric"), InferenceEngine::Exception);
}

TEST_P(IEClassGetConfigTest, GetConfigNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto configValues = p.as<std::vector<std::string>>();

    for (auto &&confKey : configValues) {
        InferenceEngine::Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.GetConfig(target_device, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(IEClassGetConfigTest, GetConfigHeteroNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto configValues = p.as<std::vector<std::string>>();

    for (auto &&confKey : configValues) {
        ASSERT_NO_THROW(ie.GetConfig(target_device, confKey));
    }
}

TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigHeteroThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_THROW(p = ie.GetConfig(CommonTestUtils::DEVICE_HETERO, "unsupported_config"), InferenceEngine::Exception);
}

TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigHeteroWithDeviceThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_THROW(p = ie.GetConfig(CommonTestUtils::DEVICE_HETERO + std::string(":") +  target_device, HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)),
                 InferenceEngine::Exception);
}

TEST_P(IEClassGetConfigTest_ThrowUnsupported, GetConfigThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_THROW(p = ie.GetConfig(target_device, "unsupported_config"), InferenceEngine::Exception);
}

TEST_P(IEClassSpecificDeviceTestGetConfig, GetConfigSpecificDeviceNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    std::string deviceID, cleartarget_device;
    auto pos =  target_device.find('.');
    if (pos != std::string::npos) {
        cleartarget_device =  target_device.substr(0, pos);
        deviceID =   target_device.substr(pos + 1,   target_device.size());
    }
    if (!supportsDeviceID(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    auto deviceIDs = ie.GetMetric(cleartarget_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL() << "Incorrect DeviceID number!" << std::endl;
    }

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto configValues = p.as<std::vector<std::string>>();

    for (auto &&confKey : configValues) {
        InferenceEngine::Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.GetConfig(target_device, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(IEClassGetAvailableDevices, GetAvailableDevicesNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    std::vector<std::string> devices;

    ASSERT_NO_THROW(devices = ie.GetAvailableDevices());

    bool deviceFound = false;
    std::cout << "Available devices: " << std::endl;
    for (auto &&device : devices) {
        if (device.find(target_device) != std::string::npos) {
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

TEST_P(IEClassQueryNetworkTest, QueryNetworkHETEROWithDeviceIDNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty())
       GTEST_FAIL() << "Incorrect DeviceID number" << std::endl;
    ASSERT_NO_THROW(ie.QueryNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_HETERO,
                                    {{"TARGET_FALLBACK",  target_device + "." + deviceIDs[0] + "," +  target_device}}));
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithDeviceID) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    ASSERT_NO_THROW(ie.QueryNetwork(simpleCnnNetwork,  target_device + ".0"));
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithBigDeviceIDThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    ASSERT_THROW(ie.QueryNetwork(actualCnnNetwork,  target_device + ".110"), InferenceEngine::Exception);
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkWithInvalidDeviceIDThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    ASSERT_THROW(ie.QueryNetwork(actualCnnNetwork,  target_device + ".l0"), InferenceEngine::Exception);
}

TEST_P(IEClassQueryNetworkTest, QueryNetworkHETEROWithBigDeviceIDThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    ASSERT_THROW(ie.QueryNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_HETERO,
                                 {{"TARGET_FALLBACK",  target_device + ".100," +  target_device}}), InferenceEngine::Exception);
}

//
// LoadNetwork
//

TEST(IEClassBasicTest, smoke_LoadNetworkToDefaultDeviceNoThrow) {
    InferenceEngine::CNNNetwork actualCnnNetwork;
    std::shared_ptr<ngraph::Function> actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(actualCnnNetwork = InferenceEngine::CNNNetwork(actualNetwork));
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork));
}

TEST_P(IEClassNetworkTestP, LoadNetworkActualNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork,  target_device));
}

TEST_P(IEClassNetworkTestP, LoadNetworkMultiWithoutSettingDevicePrioritiesThrows) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    try {
        ie.LoadNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_MULTI);
    } catch (InferenceEngine::Exception& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("KEY_MULTI_DEVICE_PRIORITIES key is not set for"),
                            error.what());
    } catch (...) {
        FAIL() << "LoadNetowrk failed for unexpected reson.";
    }
}

TEST_P(IEClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") +  target_device));
}

TEST_P(IEClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK",  target_device}}));
}

TEST_P(IEClassNetworkTestP, LoadNetworkCreateDefaultExecGraphResult) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    auto net = ie.LoadNetwork(actualCnnNetwork,  target_device);
    auto exec_function = net.GetExecGraphInfo().getFunction();
    ASSERT_NE(nullptr, exec_function);
    auto actual_parameters = exec_function->get_parameters();
    auto actual_results = exec_function->get_results();
    auto expected_parameters = actualCnnNetwork.getFunction()->get_parameters();
    auto expected_results = actualCnnNetwork.getFunction()->get_results();
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

TEST_P(IEClassLoadNetworkTestWithThrow, LoadNetworkActualWithThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ASSERT_THROW(ie.LoadNetwork(actualCnnNetwork,  target_device), InferenceEngine::Exception);
}

TEST_P(IEClassSeveralDevicesTestLoadNetwork, LoadNetworkActualSeveralDevicesNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    std::string cleartarget_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        cleartarget_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    auto deviceIDs = ie.GetMetric(cleartarget_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect DeviceID num" << std::endl;

    std::string multitarget_device = CommonTestUtils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : target_devices) {
        multitarget_device += dev_name;
        if (&dev_name != &(target_devices.back())) {
            multitarget_device += ",";
        }
    }
    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork, multitarget_device));
}

using IEClassLoadNetworkTest = IEClassQueryNetworkTest;
//
// LoadNetwork with HETERO on particular device
//
TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROWithDeviceIDNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty())
        GTEST_SKIP();
    std::string heteroDevice = CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device + "." + deviceIDs[0] + "," + target_device;
    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork, heteroDevice));
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkWithDeviceIDNoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty())
        GTEST_FAIL() << "Incorrect deviceID" << std::endl;
    ASSERT_NO_THROW(ie.LoadNetwork(simpleCnnNetwork, target_device + "." + deviceIDs[0]));
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkWithBigDeviceIDThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.LoadNetwork(actualCnnNetwork, target_device + ".10"), InferenceEngine::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkWithInvalidDeviceIDThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    ASSERT_THROW(ie.LoadNetwork(actualCnnNetwork, target_device + ".l0"), InferenceEngine::Exception);
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROWithBigDeviceIDThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    ASSERT_THROW(ie.LoadNetwork(actualCnnNetwork, "HETERO",
                                {{"TARGET_FALLBACK", target_device + ".100," + CommonTestUtils::DEVICE_CPU}}), InferenceEngine::Exception);
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROAndDeviceIDThrows) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    ASSERT_THROW(ie.LoadNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_HETERO,
                                {{"TARGET_FALLBACK",     target_device + "," + CommonTestUtils::DEVICE_CPU},
                                    {CONFIG_KEY(DEVICE_ID), "110"}}), InferenceEngine::Exception);
}

//
// LoadNetwork with HETERO on MULTI combinations particular device
//

TEST_P(IEClassLoadNetworkTest, LoadNetworkHETEROwithMULTINoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    std::string devices;
    auto availableDevices = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    for (auto &&device : availableDevices) {
        devices += target_device + '.' + device;
        if (&device != &(availableDevices.back())) {
            devices += ',';
        }
    }
    std::string targetFallback(CommonTestUtils::DEVICE_MULTI + std::string(",") + target_device);
    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_HETERO, {
            {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
            {"TARGET_FALLBACK",                   targetFallback}}));
}

TEST_P(IEClassLoadNetworkTest, LoadNetworkMULTIwithHETERONoThrow) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    std::string devices;
    auto availableDevices = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    for (auto &&device : availableDevices) {
        devices += CommonTestUtils::DEVICE_HETERO + std::string(".") + device;
        if (&device != &(availableDevices.back())) {
            devices += ',';
        }
    }
    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_MULTI, {
            {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
            {"TARGET_FALLBACK",                   target_device + "," + target_device}}));
}

//
// QueryNetwork with HETERO on MULTI combinations particular device
//

TEST_P(IEClassLoadNetworkTest, QueryNetworkHETEROWithMULTINoThrow_V10) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    std::string devices;
    auto availableDevices = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    for (auto &&device : availableDevices) {
        devices += target_device + '.' + device;
        if (&device != &(availableDevices.back())) {
            devices += ',';
        }
    }
    auto function = multinputCnnNetwork.getFunction();
    ASSERT_NE(nullptr, function);
    std::unordered_set<std::string> expectedLayers;
    for (auto &&node : function->get_ops()) {
        expectedLayers.emplace(node->get_friendly_name());
    }
    InferenceEngine::QueryNetworkResult result;
    std::string targetFallback(CommonTestUtils::DEVICE_MULTI + std::string(",") + target_device);
    ASSERT_NO_THROW(result = ie.QueryNetwork(multinputCnnNetwork, CommonTestUtils::DEVICE_HETERO, {
            {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
            {"TARGET_FALLBACK",                   targetFallback}}));

    std::unordered_set<std::string> actualLayers;
    for (auto &&layer : result.supportedLayersMap) {
        actualLayers.emplace(layer.first);
    }
    ASSERT_EQ(expectedLayers, actualLayers);
}

TEST_P(IEClassLoadNetworkTest, QueryNetworkMULTIWithHETERONoThrow_V10) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    std::string devices;
    auto availableDevices = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    for (auto &&device : availableDevices) {
        devices += "HETERO." + device;
        if (&device != &(availableDevices.back())) {
            devices += ',';
        }
    }
    auto function = multinputCnnNetwork.getFunction();
    ASSERT_NE(nullptr, function);
    std::unordered_set<std::string> expectedLayers;
    for (auto &&node : function->get_ops()) {
        expectedLayers.emplace(node->get_friendly_name());
    }
    InferenceEngine::QueryNetworkResult result;
    ASSERT_NO_THROW(result = ie.QueryNetwork(multinputCnnNetwork, CommonTestUtils::DEVICE_MULTI, {
            {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), devices},
            {"TARGET_FALLBACK",                   target_device + "," + target_device}}));

    std::unordered_set<std::string> actualLayers;
    for (auto &&layer : result.supportedLayersMap) {
        actualLayers.emplace(layer.first);
    }
    ASSERT_EQ(expectedLayers, actualLayers);
}

TEST_P(IEClassLoadNetworkAfterCoreRecreateTest, LoadAfterRecreateCoresAndPlugins) {
    InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
    {
        auto versions = ie.GetVersions(std::string(CommonTestUtils::DEVICE_MULTI) + ":" + target_device + "," + CommonTestUtils::DEVICE_CPU);
        ASSERT_EQ(3, versions.size());
    }
    std::map<std::string, std::string> config;
    if (target_device == CommonTestUtils::DEVICE_CPU) {
        config.insert({"CPU_THREADS_NUM", "3"});
    }
    ASSERT_NO_THROW({
                        InferenceEngine::Core  ie = BehaviorTestsUtils::createIECoreWithTemplate();
                        std::string name = actualCnnNetwork.getInputsInfo().begin()->first;
                        actualCnnNetwork.getInputsInfo().at(name)->setPrecision(InferenceEngine::Precision::U8);
                        auto executableNetwork = ie.LoadNetwork(actualCnnNetwork, target_device, config);
                    });
};

TEST_P(IEClassSetDefaultDeviceIDTest, SetDefaultDeviceIDNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL() << "Incorrect DeviceID" << std::endl;
    }
    std::string value;
    ASSERT_NO_THROW(ie.SetConfig({{ InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, deviceID },
                                  { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES }},
                                 target_device));
    ASSERT_NO_THROW(value = ie.GetConfig(target_device, InferenceEngine::PluginConfigParams::KEY_PERF_COUNT).as<std::string>());
    ASSERT_EQ(value, InferenceEngine::PluginConfigParams::YES);
}

TEST_P(IEClassSetGlobalConfigTest, SetGlobalConfigNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    InferenceEngine::Parameter ref, src;
    for (auto& dev_id : deviceIDs) {
        ASSERT_NO_THROW(ie.SetConfig({{ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO }},
                                     target_device + "." + dev_id));
    }
    ASSERT_NO_THROW(ie.SetConfig({{ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES }}, target_device));
    ASSERT_NO_THROW(ref = ie.GetConfig(target_device, InferenceEngine::PluginConfigParams::KEY_PERF_COUNT));

    for (auto& dev_id : deviceIDs) {
        ASSERT_NO_THROW(src = ie.GetConfig(target_device + "." + dev_id, InferenceEngine::PluginConfigParams::KEY_PERF_COUNT));
        ASSERT_EQ(src, ref);
    }
}

TEST_P(IEClassSeveralDevicesTestDefaultCore, DefaultCoreSeveralDevicesNoThrow) {
    InferenceEngine::Core ie;

    std::string cleartarget_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        cleartarget_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support DeviceID" << std::endl;
    }
    if (!supportsAvaliableDevices(ie, cleartarget_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices" << std::endl;
    }
    auto deviceIDs = ie.GetMetric(cleartarget_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect DeviceID" << std::endl;

    for (size_t i = 0; i < target_devices.size(); ++i) {
        ASSERT_NO_THROW(ie.SetConfig({{ InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, std::to_string(i + 2) }}, target_devices[i]));
    }
    std::string res;
    for (size_t i = 0; i < target_devices.size(); ++i) {
        ASSERT_NO_THROW(res = ie.GetConfig(target_devices[i], InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS).as<std::string>());
        ASSERT_EQ(res, std::to_string(i + 2));
    }
}
} // namespace BehaviorTestsDefinitions

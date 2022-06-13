// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include <openvino/runtime/properties.hpp>

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

#define OV_ASSERT_PROPERTY_SUPPORTED(property_key)                                                 \
{                                                                                                  \
    auto properties = ie.get_property(deviceName, ov::supported_properties);                       \
    auto it = std::find(properties.begin(), properties.end(), property_key);                       \
    ASSERT_NE(properties.end(), it);                                                               \
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

using DevicePriorityParams = std::tuple<
        std::string,            // Device name
        ov::AnyMap              // Configuration key and its default value
>;

class OVClassSetDevicePriorityConfigTest : public ::testing::Test, public ::testing::WithParamInterface<DevicePriorityParams> {
protected:
    std::string deviceName;
    ov::AnyMap configuration;
    std::shared_ptr<ngraph::Function> actualNetwork;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(deviceName, configuration) = GetParam();
        actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    }
};

using OVClassNetworkTestP = OVClassBaseTestP;
using OVClassQueryNetworkTest = OVClassBaseTestP;
using OVClassImportExportTestP = OVClassBaseTestP;
using OVClassGetMetricTest_SUPPORTED_METRICS = OVClassBaseTestP;
using OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassBaseTestP;
using OVClassGetMetricTest_AVAILABLE_DEVICES = OVClassBaseTestP;
using OVClassGetMetricTest_FULL_DEVICE_NAME = OVClassBaseTestP;
using OVClassGetMetricTest_FULL_DEVICE_NAME_with_DEVICE_ID = OVClassBaseTestP;
using OVClassGetMetricTest_DEVICE_UUID = OVClassBaseTestP;
using OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES = OVClassBaseTestP;
using OVClassGetMetricTest_DEVICE_GOPS = OVClassBaseTestP;
using OVClassGetMetricTest_DEVICE_TYPE = OVClassBaseTestP;
using OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS = OVClassBaseTestP;
using OVClassGetMetricTest_MAX_BATCH_SIZE = OVClassBaseTestP;
using OVClassGetMetricTest_ThrowUnsupported = OVClassBaseTestP;
using OVClassGetConfigTest = OVClassBaseTestP;
using OVClassGetConfigTest_ThrowUnsupported = OVClassBaseTestP;
using OVClassGetAvailableDevices = OVClassBaseTestP;
using OVClassGetMetricTest_RANGE_FOR_STREAMS = OVClassBaseTestP;
using OVClassLoadNetworkAfterCoreRecreateTest = OVClassBaseTestP;
using OVClassLoadNetworkTest = OVClassQueryNetworkTest;
using OVClassSetGlobalConfigTest = OVClassBaseTestP;
using OVClassSetModelPriorityConfigTest = OVClassBaseTestP;
using OVClassSetLogLevelConfigTest = OVClassBaseTestP;
using OVClassSpecificDeviceTestSetConfig = OVClassBaseTestP;
using OVClassSpecificDeviceTestGetConfig = OVClassBaseTestP;
using OVClassLoadNetworkWithCorrectPropertiesTest = OVClassSetDevicePriorityConfigTest;
using OVClassLoadNetworkWithIncorrectPropertiesTest = OVClassSetDevicePriorityConfigTest;

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

inline bool supportsAvaliableDevices(ov::Core& ie, const std::string& deviceName) {
    auto supported_properties = ie.get_property(deviceName, ov::supported_properties);
    return supported_properties.end() !=
           std::find(std::begin(supported_properties), std::end(supported_properties), ov::available_devices);
}

bool supportsDeviceID(ov::Core& ie, const std::string& deviceName) {
    auto supported_properties =
            ie.get_property(deviceName, ov::supported_properties);
    return supported_properties.end() !=
           std::find(std::begin(supported_properties), std::end(supported_properties), ov::device::id);
}

TEST(OVClassBasicTest, smoke_createDefault) {
    OV_ASSERT_NO_THROW(ov::Core ie);
}

TEST_P(OVClassBasicTestP, registerExistingPluginThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.register_plugin(pluginName, deviceName), ov::Exception);
}

// TODO: CVS-68982
#ifndef OPENVINO_STATIC_LIBRARY

TEST_P(OVClassBasicTestP, registerNewPluginNoThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.register_plugin(pluginName, "NEW_DEVICE_NAME"));
    OV_ASSERT_NO_THROW(ie.get_property("NEW_DEVICE_NAME", ov::supported_properties));
}

TEST(OVClassBasicTest, smoke_registerExistingPluginFileThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.register_plugins("nonExistPlugins.xml"), ov::Exception);
    ASSERT_THROW(ie.register_plugins("nonExistPlugins.xml"), ov::Exception);
}

TEST(OVClassBasicTest, smoke_createNonExistingConfigThrows) {
    ASSERT_THROW(ov::Core ie("nonExistPlugins.xml"), ov::Exception);
}

#ifdef __linux__

TEST(OVClassBasicTest, smoke_createMockEngineConfigNoThrows) {
    std::string filename{"mock_engine_valid.xml"};
    std::string content{"<ie><plugins><plugin name=\"mock\" location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    OV_ASSERT_NO_THROW(ov::Core ie(filename));
    CommonTestUtils::removeFile(filename.c_str());
}

TEST(OVClassBasicTest, smoke_createMockEngineConfigThrows) {
    std::string filename{"mock_engine.xml"};
    std::string content{"<ie><plugins><plugin location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_THROW(ov::Core ie(filename), ov::Exception);
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

            ov::Core ie = createCoreWithTemplate();
            GTEST_COUT << "Core created " << testIndex << std::endl;
            OV_ASSERT_NO_THROW(ie.register_plugins(::ov::util::wstring_to_string(pluginsXmlW)));
            CommonTestUtils::removeFile(pluginsXmlW);
#    if defined __linux__ && !defined(__APPLE__)
            OV_ASSERT_NO_THROW(ie.get_versions("mock"));  // from pluginXML
#    endif
            OV_ASSERT_NO_THROW(ie.get_versions(deviceName));
            GTEST_COUT << "Plugin created " << testIndex << std::endl;

            OV_ASSERT_NO_THROW(ie.register_plugin(pluginName, "TEST_DEVICE"));
            OV_ASSERT_NO_THROW(ie.get_versions("TEST_DEVICE"));
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
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.get_versions(deviceName + ".0"));
}

TEST_P(OVClassBasicTestP, getVersionsByDeviceClassNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.get_versions(deviceName));
}

TEST_P(OVClassBasicTestP, getVersionsNonEmpty) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_EQ(2, ie.get_versions(CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName).size());
}

//
// UnregisterPlugin
//

TEST_P(OVClassBasicTestP, unregisterExistingPluginNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    // device instance is not created yet
    ASSERT_THROW(ie.unload_plugin(deviceName), ov::Exception);

    // make the first call to IE which created device instance
    ie.get_versions(deviceName);
    // now, we can unregister device
    OV_ASSERT_NO_THROW(ie.unload_plugin(deviceName));
}

TEST_P(OVClassBasicTestP, accessToUnregisteredPluginThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.unload_plugin(deviceName), ov::Exception);
    OV_ASSERT_NO_THROW(ie.get_versions(deviceName));
    OV_ASSERT_NO_THROW(ie.unload_plugin(deviceName));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::AnyMap{}));
    OV_ASSERT_NO_THROW(ie.get_versions(deviceName));
    OV_ASSERT_NO_THROW(ie.unload_plugin(deviceName));
}

TEST(OVClassBasicTest, smoke_unregisterNonExistingPluginThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.unload_plugin("unkown_device"), ov::Exception);
}

//
// SetConfig
//

TEST_P(OVClassBasicTestP, SetConfigAllThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property({{"unsupported_key", "4"}}));
    ASSERT_ANY_THROW(ie.get_versions(deviceName));
}

TEST_P(OVClassBasicTestP, SetConfigForUnRegisteredDeviceThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.set_property("unregistered_device", {{"unsupported_key", "4"}}), ov::Exception);
}

TEST_P(OVClassBasicTestP, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::enable_profiling(true)));
}

TEST_P(OVClassBasicTestP, SetConfigAllNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(ov::enable_profiling(true)));
    OV_ASSERT_NO_THROW(ie.get_versions(deviceName));
}

TEST(OVClassBasicTest, smoke_SetConfigHeteroThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_HETERO, ov::enable_profiling(true)));
}

TEST_P(OVClassBasicTestP, SetConfigHeteroTargetFallbackThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities(deviceName)));
}

TEST_P(OVClassBasicTestP, smoke_SetConfigHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string value;

    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities(deviceName)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities));
    ASSERT_EQ(deviceName, value);

    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities(deviceName)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities));
    ASSERT_EQ(deviceName, value);
}

TEST(OVClassBasicTest, smoke_SetConfigAutoNoThrows) {
    ov::Core ie = createCoreWithTemplate();

    // priority config test
    ov::hint::Priority value;
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::LOW)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::LOW);
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::MEDIUM)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::MEDIUM);
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::HIGH)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::HIGH);
}

TEST_P(OVClassSpecificDeviceTestSetConfig, SetConfigSpecificDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::string deviceID, clearDeviceName;
    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceName.substr(0, pos);
        deviceID =  deviceName.substr(pos + 1,  deviceName.size());
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    auto deviceIDs = ie.get_property(clearDeviceName, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_SKIP();
    }

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::enable_profiling(true)));
    bool value = false;
    OV_ASSERT_NO_THROW(value = ie.get_property(deviceName, ov::enable_profiling));
    ASSERT_TRUE(value);
}

TEST_P(OVClassSetModelPriorityConfigTest, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    // priority config test
    ov::hint::Priority value;
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::model_priority(ov::hint::Priority::LOW)));
    OV_ASSERT_NO_THROW(value = ie.get_property(deviceName, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::LOW);
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::model_priority(ov::hint::Priority::MEDIUM)));
    OV_ASSERT_NO_THROW(value = ie.get_property(deviceName, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::MEDIUM);
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::model_priority(ov::hint::Priority::HIGH)));
    OV_ASSERT_NO_THROW(value = ie.get_property(deviceName, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::HIGH);
}

TEST_P(OVClassSetDevicePriorityConfigTest, SetConfigAndCheckGetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string devicePriority;
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, configuration));
    OV_ASSERT_NO_THROW(devicePriority = ie.get_property(deviceName, ov::device::priorities));
    ASSERT_EQ(devicePriority, configuration[ov::device::priorities.name()].as<std::string>());
}

TEST_P(OVClassSetLogLevelConfigTest, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    // log level
    ov::log::Level logValue;
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::log::level(ov::log::Level::NO)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(deviceName, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::NO);
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::log::level(ov::log::Level::ERR)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(deviceName, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::ERR);
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::log::level(ov::log::Level::WARNING)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(deviceName, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::WARNING);
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::log::level(ov::log::Level::INFO)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(deviceName, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::INFO);
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::log::level(ov::log::Level::DEBUG)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(deviceName, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::DEBUG);
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::log::level(ov::log::Level::TRACE)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(deviceName, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::TRACE);
}
//
// QueryNetwork
//

TEST_P(OVClassNetworkTestP, QueryNetworkActualThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.query_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(OVClassNetworkTestP, QueryNetworkActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    try {
        ie.query_model(actualNetwork, deviceName);
    } catch (const ov::Exception& ex) {
        std::string message = ex.what();
        ASSERT_STR_CONTAINS(message, "[NOT_IMPLEMENTED]  ngraph::Function is not supported natively");
    }
}

TEST_P(OVClassNetworkTestP, QueryNetworkWithKSO) {
    ov::Core ie = createCoreWithTemplate();

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
    ov::Core ie = createCoreWithTemplate();

    std::string clearDeviceName;
    auto pos = deviceNames.begin()->find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceNames.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    auto deviceIDs = ie.get_property(clearDeviceName, ov::available_devices);
    if (deviceIDs.size() < deviceNames.size())
        GTEST_SKIP();

    std::string multiDeviceName = CommonTestUtils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : deviceNames) {
        multiDeviceName += dev_name;
        if (&dev_name != &(deviceNames.back())) {
            multiDeviceName += ",";
        }
    }
    OV_ASSERT_NO_THROW(ie.query_model(actualNetwork, multiDeviceName));
}

TEST_P(OVClassNetworkTestP, SetAffinityWithConstantBranches) {
    ov::Core ie = createCoreWithTemplate();

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
    ov::Core ie = createCoreWithTemplate();

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
    ov::Core ie = createCoreWithTemplate();
    ov::SupportedOpsMap res;
    OV_ASSERT_NO_THROW(
        res = ie.query_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, ov::device::priorities(deviceName)));
    ASSERT_LT(0, res.size());
}

TEST_P(OVClassNetworkTestP, QueryNetworkMultiThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.query_model(actualNetwork, CommonTestUtils::DEVICE_MULTI), ov::Exception);
}

TEST(OVClassBasicTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string deviceName = CommonTestUtils::DEVICE_HETERO;

    std::vector<ov::PropertyName> t;
    OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::supported_properties));

    std::cout << "Supported HETERO properties: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << " is_mutable: " << str.is_mutable() << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

TEST(OVClassBasicTest, smoke_GetMetricSupportedConfigKeysHeteroThrows) {
    ov::Core ie = createCoreWithTemplate();
    // TODO: check
    std::string targetDevice = CommonTestUtils::DEVICE_HETERO + std::string(":") + CommonTestUtils::DEVICE_CPU;
    ASSERT_THROW(ie.get_property(targetDevice, ov::supported_properties), ov::Exception);
}

TEST_P(OVClassGetMetricTest_SUPPORTED_METRICS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> t;

    OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::supported_properties));

    std::cout << "Supported properties: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << " is_mutable: " << str.is_mutable() << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> t;

    OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::supported_properties));

    std::cout << "Supported config values: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << " is_mutable: " << str.is_mutable() << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassGetMetricTest_AVAILABLE_DEVICES, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<std::string> t;

    OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::available_devices));

    std::cout << "Available devices: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::available_devices);
}

TEST_P(OVClassGetMetricTest_FULL_DEVICE_NAME, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string t;

    OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::device::full_name));
    std::cout << "Full device name: " << std::endl << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::full_name);
}

TEST_P(OVClassGetMetricTest_FULL_DEVICE_NAME_with_DEVICE_ID, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string t;

    if (supportsDeviceID(ie, deviceName)) {
        auto device_ids = ie.get_property(deviceName, ov::available_devices);
        ASSERT_GT(device_ids.size(), 0);
        OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::device::full_name, ov::device::id(device_ids.front())));
        std::cout << "Device " << device_ids.front() << " " <<  ", Full device name: " << std::endl << t << std::endl;
        OV_ASSERT_PROPERTY_SUPPORTED(ov::device::full_name);
    } else {
        GTEST_SKIP() << "Device id is not supported";
    }
}

TEST_P(OVClassGetMetricTest_DEVICE_UUID, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ov::device::UUID t;

    OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::device::uuid));
    std::cout << "Device uuid: " << std::endl << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::uuid);
}

TEST_P(OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<std::string> t;
    OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::device::capabilities));
    std::cout << "Optimization capabilities: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }
    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::capabilities);
}

TEST_P(OVClassGetMetricTest_MAX_BATCH_SIZE, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    uint32_t max_batch_size = 0;

    ASSERT_NO_THROW(max_batch_size = ie.get_property(deviceName, ov::max_batch_size));

    std::cout << "Max batch size: " << max_batch_size << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::max_batch_size);
}

TEST_P(OVClassGetMetricTest_DEVICE_GOPS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::cout << "Device GOPS: " << std::endl;
    for (auto&& kv : ie.get_property(deviceName, ov::device::gops)) {
        std::cout << kv.first << ": " << kv.second << std::endl;
    }
    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::gops);
}

TEST_P(OVClassGetMetricTest_DEVICE_TYPE, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::type);
    ov::device::Type t = {};
    OV_ASSERT_NO_THROW(t = ie.get_property(deviceName, ov::device::type));
    std::cout << "Device Type: " << t << std::endl;
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    unsigned int start{0}, end{0}, step{0};

    ASSERT_NO_THROW(std::tie(start, end, step) = ie.get_property(deviceName, ov::range_for_async_infer_requests));

    std::cout << "Range for async infer requests: " << std::endl
    << start << std::endl
    << end << std::endl
    << step << std::endl
    << std::endl;

    ASSERT_LE(start, end);
    ASSERT_GE(step, 1);
    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_async_infer_requests);
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_STREAMS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    unsigned int start = 0, end = 0;

    ASSERT_NO_THROW(std::tie(start, end) = ie.get_property(deviceName, ov::range_for_streams));

    std::cout << "Range for streams: " << std::endl
    << start << std::endl
    << end << std::endl
    << std::endl;

    ASSERT_LE(start, end);
    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_streams);
}

TEST_P(OVClassGetMetricTest_ThrowUnsupported, GetMetricThrow) {
    ov::Core ie = createCoreWithTemplate();

    ASSERT_THROW(ie.get_property(deviceName, "unsupported_metric"), ov::Exception);
}

TEST_P(OVClassGetConfigTest, GetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> configValues;

    OV_ASSERT_NO_THROW(configValues = ie.get_property(deviceName, ov::supported_properties));

    for (auto&& confKey : configValues) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = ie.get_property(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassGetConfigTest, GetConfigHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> configValues;
    OV_ASSERT_NO_THROW(configValues = ie.get_property(deviceName, ov::supported_properties));

    for (auto&& confKey : configValues) {
        OV_ASSERT_NO_THROW(ie.get_property(deviceName, confKey));
    }
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroThrow) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.get_property(CommonTestUtils::DEVICE_HETERO, "unsupported_config"), ov::Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroWithDeviceThrow) {
    ov::Core ie = createCoreWithTemplate();

    ASSERT_THROW(ie.get_property(CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName,
                                   ov::device::priorities),
                 ov::Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigThrow) {
    ov::Core ie = createCoreWithTemplate();

    ASSERT_THROW(ie.get_property(deviceName, "unsupported_config"), ov::Exception);
}

TEST_P(OVClassSpecificDeviceTestGetConfig, GetConfigSpecificDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
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
    auto deviceIDs = ie.get_property(clearDeviceName, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_SKIP();
    }

    std::vector<ov::PropertyName> configValues;
    OV_ASSERT_NO_THROW(configValues = ie.get_property(deviceName, ov::supported_properties));

    for (auto &&confKey : configValues) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = ie.get_property(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassGetAvailableDevices, GetAvailableDevicesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<std::string> devices;

    OV_ASSERT_NO_THROW(devices = ie.get_available_devices());

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
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.get_property(deviceName, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_SKIP();
        OV_ASSERT_NO_THROW(ie.query_model(actualNetwork,
                                          CommonTestUtils::DEVICE_HETERO,
                                          ov::device::priorities(deviceName + "." + deviceIDs[0], deviceName)));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithDeviceID) {
    ov::Core ie = createCoreWithTemplate();

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
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork, deviceName + ".110"), ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithInvalidDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork, deviceName + ".l0"), ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.query_model(actualNetwork,
                                    CommonTestUtils::DEVICE_HETERO,
                                    ov::device::priorities(deviceName + ".100", deviceName)),
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
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, deviceName));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, ov::device::priorities(deviceName)));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceUsingDevicePropertiesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork,
        CommonTestUtils::DEVICE_HETERO,
        ov::device::priorities(deviceName),
        ov::device::properties(deviceName,
            ov::enable_profiling(true))));
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
    ov::Core ie = createCoreWithTemplate();

    std::string clearDeviceName;
    auto pos = deviceNames.begin()->find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceNames.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    auto deviceIDs = ie.get_property(clearDeviceName, ov::available_devices);
    if (deviceIDs.size() < deviceNames.size())
        GTEST_SKIP();

    std::string multiDeviceName = CommonTestUtils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : deviceNames) {
        multiDeviceName += dev_name;
        if (&dev_name != &(deviceNames.back())) {
            multiDeviceName += ",";
        }
    }
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, multiDeviceName));
}

//
// LoadNetwork with HETERO on particular device
//
TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.get_property(deviceName, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_SKIP();
        std::string heteroDevice =
                CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName + "." + deviceIDs[0] + "," + deviceName;
        OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, heteroDevice));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.get_property(deviceName, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_SKIP();
        OV_ASSERT_NO_THROW(ie.compile_model(simpleNetwork, deviceName + "." + deviceIDs[0]));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, deviceName + ".10"), ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkWithCorrectPropertiesTest, LoadNetworkWithCorrectPropertiesTest) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, deviceName, configuration));
}

TEST_P(OVClassLoadNetworkWithIncorrectPropertiesTest, LoadNetworkWithCorrectPropertiesTest) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.compile_model(actualNetwork, deviceName, configuration), ov::Exception);
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithInvalidDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, deviceName + ".l0"), ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                      "HETERO",
                                       ov::device::priorities(deviceName + ".100", CommonTestUtils::DEVICE_CPU)),
                     ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROAndDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                      CommonTestUtils::DEVICE_HETERO,
                                      ov::device::priorities(deviceName, CommonTestUtils::DEVICE_CPU),
                                      ov::device::id("110")),
                     ov::Exception);
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with AUTO on MULTI combinations particular device
//
TEST_P(OVClassLoadNetworkTest, LoadNetworkMULTIwithAUTONoThrow) {
    ov::Core ie = createCoreWithTemplate();
    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_property(deviceName, ov::available_devices);
        for (auto&& device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        OV_ASSERT_NO_THROW(
            ie.compile_model(actualNetwork,
                             CommonTestUtils::DEVICE_MULTI,
                             ov::device::properties(CommonTestUtils::DEVICE_AUTO, ov::device::priorities(devices)),
                             ov::device::properties(CommonTestUtils::DEVICE_MULTI,
                                                    ov::device::priorities(CommonTestUtils::DEVICE_AUTO, deviceName))));
    } else {
        GTEST_SKIP();
    }
}

//
// LoadNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROwithMULTINoThrow) {
    ov::Core ie = createCoreWithTemplate();
    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_property(deviceName, ov::available_devices);
        for (auto&& device : availableDevices) {
            devices += deviceName + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        OV_ASSERT_NO_THROW(
            ie.compile_model(actualNetwork,
                             CommonTestUtils::DEVICE_HETERO,
                             ov::device::properties(CommonTestUtils::DEVICE_MULTI,
                                               ov::device::priorities(devices)),
                             ov::device::properties(CommonTestUtils::DEVICE_HETERO,
                                               ov::device::priorities(CommonTestUtils::DEVICE_MULTI, deviceName))));
    } else {
        GTEST_SKIP();
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkMULTIwithHETERONoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_property(deviceName, ov::available_devices);
        for (auto&& device : availableDevices) {
            devices += CommonTestUtils::DEVICE_HETERO + std::string(".") + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        OV_ASSERT_NO_THROW(ie.compile_model(
            actualNetwork,
            CommonTestUtils::DEVICE_MULTI,
            ov::device::properties(CommonTestUtils::DEVICE_MULTI, ov::device::priorities(devices)),
            ov::device::properties(CommonTestUtils::DEVICE_HETERO, ov::device::priorities(deviceName, deviceName))));
    } else {
        GTEST_SKIP();
    }
}

//
// QueryNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, QueryNetworkHETEROWithMULTINoThrow_V10) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_property(deviceName, ov::available_devices);
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
        ov::SupportedOpsMap result;
        std::string hetero_device_priorities(CommonTestUtils::DEVICE_MULTI + std::string(",") + deviceName);
        OV_ASSERT_NO_THROW(result = ie.query_model(
                            multinputNetwork,
                            CommonTestUtils::DEVICE_HETERO,
                            ov::device::properties(CommonTestUtils::DEVICE_MULTI,
                                                   ov::device::priorities(devices)),
                            ov::device::properties(CommonTestUtils::DEVICE_HETERO,
                                                   ov::device::priorities(CommonTestUtils::DEVICE_MULTI,
                                                                       deviceName))));

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
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, deviceName) && supportsAvaliableDevices(ie, deviceName)) {
        std::string devices;
        auto availableDevices = ie.get_property(deviceName, ov::available_devices);
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
        ov::SupportedOpsMap result;
        OV_ASSERT_NO_THROW(result = ie.query_model(multinputNetwork,
                                                 CommonTestUtils::DEVICE_MULTI,
                                                 ov::device::properties(CommonTestUtils::DEVICE_MULTI,
                                                                    ov::device::priorities(devices)),
                                                 ov::device::properties(CommonTestUtils::DEVICE_HETERO,
                                                                    ov::device::priorities(deviceName, deviceName))));

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
    ov::Core ie = createCoreWithTemplate();
    {
        auto versions = ie.get_versions(std::string(CommonTestUtils::DEVICE_MULTI) + ":" + deviceName + "," +
                                        CommonTestUtils::DEVICE_CPU);
        ASSERT_EQ(3, versions.size());
    }
    ov::AnyMap config;
    if (deviceName == CommonTestUtils::DEVICE_CPU) {
        config.insert(ov::enable_profiling(true));
    }
    // OV_ASSERT_NO_THROW({
    //     ov::Core ie = createCoreWithTemplate();
    //     std::string name = actualNetwork.getInputsInfo().begin()->first;
    //     actualNetwork.getInputsInfo().at(name)->setPrecision(Precision::U8);
    //     auto executableNetwork = ie.compile_model(actualNetwork, deviceName, config);
    // });
};

TEST_P(OVClassSetDefaultDeviceIDTest, SetDefaultDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto deviceIDs = ie.get_property(deviceName, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_SKIP();
    }
    std::string value;
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::device::id(deviceID), ov::enable_profiling(true)));
    ASSERT_TRUE(ie.get_property(deviceName, ov::enable_profiling));
    OV_ASSERT_NO_THROW(value = ie.get_property(deviceName, ov::enable_profiling.name()).as<std::string>());
    ASSERT_EQ(value, "YES");
}

TEST_P(OVClassSetGlobalConfigTest, SetGlobalConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto deviceIDs = ie.get_property(deviceName, ov::available_devices);
    ov::Any ref, src;
    for (auto& dev_id : deviceIDs) {
        OV_ASSERT_NO_THROW(ie.set_property(deviceName + "." + dev_id, ov::enable_profiling(false)));
    }
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::enable_profiling(true)));
    OV_ASSERT_NO_THROW(ref = ie.get_property(deviceName, ov::enable_profiling.name()));

    for (auto& dev_id : deviceIDs) {
        OV_ASSERT_NO_THROW(src = ie.get_property(deviceName + "." + dev_id, ov::enable_profiling.name()));
        ASSERT_EQ(src, ref);
    }
}

TEST_P(OVClassSeveralDevicesTestDefaultCore, DefaultCoreSeveralDevicesNoThrow) {
    ov::Core ie;

    std::string clearDeviceName;
    auto pos = deviceNames.begin()->find('.');
    if (pos != std::string::npos) {
        clearDeviceName = deviceNames.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clearDeviceName) || !supportsAvaliableDevices(ie, clearDeviceName)) {
        GTEST_SKIP();
    }
    auto deviceIDs = ie.get_property(clearDeviceName, ov::available_devices);
    if (deviceIDs.size() < deviceNames.size())
        GTEST_SKIP();

    for (size_t i = 0; i < deviceNames.size(); ++i) {
        OV_ASSERT_NO_THROW(ie.set_property(deviceNames[i], ov::enable_profiling(true)));
    }
    bool res;
    for (size_t i = 0; i < deviceNames.size(); ++i) {
        OV_ASSERT_NO_THROW(res = ie.get_property(deviceNames[i], ov::enable_profiling));
        ASSERT_TRUE(res);
    }
}
}  // namespace behavior
}  // namespace test
}  // namespace ov

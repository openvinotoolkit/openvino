// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include <openvino/runtime/properties.hpp>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/util/file_util.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>
#    include "openvino/pass/manager.hpp"
#endif

namespace ov {
namespace test {
namespace behavior {

#define OV_ASSERT_PROPERTY_SUPPORTED(property_key)                                                 \
{                                                                                                  \
    auto properties = ie.get_property(target_device, ov::supported_properties);                       \
    auto it = std::find(properties.begin(), properties.end(), property_key);                       \
    ASSERT_NE(properties.end(), it);                                                               \
}

inline bool supportsAvailableDevices(ov::Core& ie, const std::string& target_device) {
    auto supported_properties = ie.get_property(target_device, ov::supported_properties);
    return supported_properties.end() !=
           std::find(std::begin(supported_properties), std::end(supported_properties), ov::available_devices);
}

inline bool supportsDeviceID(ov::Core& ie, const std::string& target_device) {
    auto supported_properties =
            ie.get_property(target_device, ov::supported_properties);
    return supported_properties.end() !=
           std::find(std::begin(supported_properties), std::end(supported_properties), ov::device::id);
}

class OVClassBasicTestP : public OVPluginTestBase,
                          public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string pluginName;

public:
    void SetUp() override {
        std::tie(pluginName, target_device) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        pluginName += OV_BUILD_POSTFIX;
        if (pluginName == (std::string("openvino_template_plugin") + OV_BUILD_POSTFIX)) {
            pluginName = ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(), pluginName);
        }
    }
};

class OVClassSetDefaultDeviceIDTest : public OVPluginTestBase,
                                      public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string deviceID;

public:
    void SetUp() override {
        std::tie(target_device, deviceID) = GetParam();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
    }
};

using DevicePriorityParams = std::tuple<
        std::string,            // Device name
        ov::AnyMap              // Configuration key and its default value
>;

class OVClassSetDevicePriorityConfigTest : public OVPluginTestBase,
                                           public ::testing::WithParamInterface<DevicePriorityParams> {
protected:
    std::string deviceName;
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> actualNetwork;

public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
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
using OVClassGetMetricTest_DEVICE_LUID = OVClassBaseTestP;
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
using OVClassSetExecutionModeHintConfigTest = OVClassBaseTestP;
using OVClassSetEnableCpuPinningHintConfigTest = OVClassBaseTestP;
using OVClassSetSchedulingCoreTypeHintConfigTest = OVClassBaseTestP;
using OVClassSetEnableHyperThreadingHintConfigTest = OVClassBaseTestP;
using OVClassSetTBBForceTerminatePropertyTest = OVClassBaseTestP;
using OVClassSetLogLevelConfigTest = OVClassBaseTestP;
using OVClassSpecificDeviceTestSetConfig = OVClassBaseTestP;
using OVClassSpecificDeviceTestGetConfig = OVClassBaseTestP;
using OVClassLoadNetworkWithCorrectPropertiesTest = OVClassSetDevicePriorityConfigTest;
using OVClassLoadNetworkWithCondidateDeviceListContainedMetaPluginTest = OVClassSetDevicePriorityConfigTest;
using OVClassLoadNetWorkReturnDefaultHintTest = OVClassSetDevicePriorityConfigTest;
using OVClassLoadNetWorkDoNotReturnDefaultHintTest = OVClassSetDevicePriorityConfigTest;
using OVClassLoadNetworkAndCheckSecondaryPropertiesTest = OVClassSetDevicePriorityConfigTest;

class OVClassSeveralDevicesTest : public OVPluginTestBase,
                                  public OVClassNetworkTest,
                                  public ::testing::WithParamInterface<std::vector<std::string>> {
public:
    std::vector<std::string> target_devices;

    void SetUp() override {
        target_device = ov::test::utils::DEVICE_MULTI;
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
        target_devices = GetParam();
    }
};

using OVClassSeveralDevicesTestLoadNetwork = OVClassSeveralDevicesTest;
using OVClassSeveralDevicesTestQueryNetwork = OVClassSeveralDevicesTest;
using OVClassSeveralDevicesTestDefaultCore = OVClassSeveralDevicesTest;

TEST(OVClassBasicTest, smoke_createDefault) {
    OV_ASSERT_NO_THROW(ov::Core ie);
}

TEST_P(OVClassBasicTestP, registerExistingPluginThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.register_plugin(pluginName, target_device), ov::Exception);
}

// TODO: CVS-68982
#ifndef OPENVINO_STATIC_LIBRARY

TEST_P(OVClassBasicTestP, registerNewPluginNoThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.register_plugin(pluginName, "NEW_DEVICE_NAME"));
    OV_ASSERT_NO_THROW(ie.get_property("NEW_DEVICE_NAME", ov::supported_properties));
}

TEST(OVClassBasicTest, smoke_registerNonExistingPluginFileThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.register_plugins("nonExistPlugins.xml"), ov::Exception);
}

TEST(OVClassBasicTest, smoke_createNonExistingConfigThrows) {
    ASSERT_THROW(ov::Core ie("nonExistPlugins.xml"), ov::Exception);
}

inline std::string getPluginFile() {
    std::string filePostfix{"mock_engine_valid.xml"};
    std::string filename = ov::test::utils::generateTestFilePrefix() + "_" + filePostfix;
    std::ostringstream stream;
    stream << "<ie><plugins><plugin name=\"mock\" location=\"";
    stream << ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
        std::string("mock_engine") + OV_BUILD_POSTFIX);
    stream << "\"></plugin></plugins></ie>";
    ov::test::utils::createFile(filename, stream.str());
    return filename;
}

TEST(OVClassBasicTest, smoke_createMockEngineConfigNoThrows) {
    const std::string filename = getPluginFile();
    OV_ASSERT_NO_THROW(ov::Core ie(filename));
    ov::test::utils::removeFile(filename.c_str());
}

TEST(OVClassBasicTest, smoke_createMockEngineConfigThrows) {
    std::string filename = ov::test::utils::generateTestFilePrefix() + "_mock_engine.xml";
    std::string content{"<ie><plugins><plugin location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    ov::test::utils::createFile(filename, content);
    ASSERT_THROW(ov::Core ie(filename), ov::Exception);
    ov::test::utils::removeFile(filename.c_str());
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
inline void generateModelFile() {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>("test_model.xml", "test_model.bin");
    auto function = ngraph::builder::subgraph::makeConvPoolReluNoReshapes({1, 3, 227, 227});
    manager.run_passes(function);
}

TEST(OVClassBasicTest, compile_model_no_property_unicode) {
    std::string model_xml_name = "test_model.xml";
    std::string model_bin_name = "test_model.bin";
    generateModelFile();
    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix = L"_" + ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring modelXmlPathW = ov::test::utils::addUnicodePostfixToPath(model_xml_name, postfix);
        std::wstring modelBinPathW = ov::test::utils::addUnicodePostfixToPath(model_bin_name, postfix);
        GTEST_COUT << testIndex << ": " << ::ov::util::wstring_to_string(modelXmlPathW) << std::endl;

        try {
            bool is_copy_successfully;
            is_copy_successfully = ov::test::utils::copyFile(model_xml_name, modelXmlPathW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << model_xml_name << "' to '"
                       << ::ov::util::wstring_to_string(modelXmlPathW) << "'";
            }

            is_copy_successfully = ov::test::utils::copyFile(model_bin_name, modelBinPathW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << model_bin_name << "' to '"
                       << ::ov::util::wstring_to_string(modelBinPathW) << "'";
            }

            ov::Core core = createCoreWithTemplate();

            OV_ASSERT_NO_THROW(core.compile_model(modelXmlPathW));
            ov::test::utils::removeFile(modelXmlPathW);
            ov::test::utils::removeFile(modelBinPathW);
            GTEST_COUT << "OK" << std::endl;
        } catch (const ov::Exception& e_next) {
            ov::test::utils::removeFile(modelXmlPathW);
            ov::test::utils::removeFile(modelBinPathW);
            ov::test::utils::removeFile(model_xml_name);
            ov::test::utils::removeFile(model_bin_name);
            FAIL() << e_next.what();
        }
    }
    ov::test::utils::removeFile(model_xml_name);
    ov::test::utils::removeFile(model_bin_name);
}

TEST(OVClassBasicTest, compile_model_with_property_unicode) {
    std::string model_xml_name = "test_model.xml";
    std::string model_bin_name = "test_model.bin";
    generateModelFile();
    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix = L"_" + ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring modelXmlPathW = ov::test::utils::addUnicodePostfixToPath(model_xml_name, postfix);
        std::wstring modelBinPathW = ov::test::utils::addUnicodePostfixToPath(model_bin_name, postfix);
        GTEST_COUT << testIndex << ": " << ::ov::util::wstring_to_string(modelXmlPathW) << std::endl;

        try {
            bool is_copy_successfully;
            is_copy_successfully = ov::test::utils::copyFile(model_xml_name, modelXmlPathW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << model_xml_name << "' to '"
                       << ::ov::util::wstring_to_string(modelXmlPathW) << "'";
            }

            is_copy_successfully = ov::test::utils::copyFile(model_bin_name, modelBinPathW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << model_bin_name << "' to '"
                       << ::ov::util::wstring_to_string(modelBinPathW) << "'";
            }

            ov::Core core = createCoreWithTemplate();

            OV_ASSERT_NO_THROW(
                core.compile_model(modelXmlPathW, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));
            ov::test::utils::removeFile(modelXmlPathW);
            ov::test::utils::removeFile(modelBinPathW);
            GTEST_COUT << "OK" << std::endl;
        } catch (const ov::Exception& e_next) {
            ov::test::utils::removeFile(modelXmlPathW);
            ov::test::utils::removeFile(modelBinPathW);
            ov::test::utils::removeFile(model_xml_name);
            ov::test::utils::removeFile(model_bin_name);
            FAIL() << e_next.what();
        }
    }
    ov::test::utils::removeFile(model_xml_name);
    ov::test::utils::removeFile(model_bin_name);
}

TEST_P(OVClassBasicTestP, compile_model_with_device_no_property_unicode) {
    std::string model_xml_name = "test_model.xml";
    std::string model_bin_name = "test_model.bin";
    generateModelFile();
    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix = L"_" + ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring modelXmlPathW = ov::test::utils::addUnicodePostfixToPath(model_xml_name, postfix);
        std::wstring modelBinPathW = ov::test::utils::addUnicodePostfixToPath(model_bin_name, postfix);
        GTEST_COUT << testIndex << ": " << ::ov::util::wstring_to_string(modelXmlPathW) << std::endl;
        try {
            bool is_copy_successfully;
            is_copy_successfully = ov::test::utils::copyFile(model_xml_name, modelXmlPathW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << model_xml_name << "' to '"
                       << ::ov::util::wstring_to_string(modelXmlPathW) << "'";
            }

            is_copy_successfully = ov::test::utils::copyFile(model_bin_name, modelBinPathW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << model_bin_name << "' to '"
                       << ::ov::util::wstring_to_string(modelBinPathW) << "'";
            }

            ov::Core core = createCoreWithTemplate();

            OV_ASSERT_NO_THROW(core.compile_model(modelXmlPathW, target_device));
            ov::test::utils::removeFile(modelXmlPathW);
            ov::test::utils::removeFile(modelBinPathW);
            GTEST_COUT << "OK" << std::endl;
        } catch (const ov::Exception& e_next) {
            ov::test::utils::removeFile(modelXmlPathW);
            ov::test::utils::removeFile(modelBinPathW);
            ov::test::utils::removeFile(model_xml_name);
            ov::test::utils::removeFile(model_bin_name);
            FAIL() << e_next.what();
        }
    }
    ov::test::utils::removeFile(model_xml_name);
    ov::test::utils::removeFile(model_bin_name);
}

TEST_P(OVClassBasicTestP, compile_model_with_device_with_property_unicode) {
    std::string model_xml_name = "test_model.xml";
    std::string model_bin_name = "test_model.bin";
    generateModelFile();
    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix = L"_" + ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring modelXmlPathW = ov::test::utils::addUnicodePostfixToPath(model_xml_name, postfix);
        std::wstring modelBinPathW = ov::test::utils::addUnicodePostfixToPath(model_bin_name, postfix);
        GTEST_COUT << testIndex << ": " << ::ov::util::wstring_to_string(modelXmlPathW) << std::endl;

        try {
            bool is_copy_successfully;
            is_copy_successfully = ov::test::utils::copyFile(model_xml_name, modelXmlPathW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << model_xml_name << "' to '"
                       << ::ov::util::wstring_to_string(modelXmlPathW) << "'";
            }

            is_copy_successfully = ov::test::utils::copyFile(model_bin_name, modelBinPathW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << model_bin_name << "' to '"
                       << ::ov::util::wstring_to_string(modelBinPathW) << "'";
            }

            ov::Core core = createCoreWithTemplate();

            OV_ASSERT_NO_THROW(core.compile_model(modelXmlPathW,
                                                  target_device,
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));
            ov::test::utils::removeFile(modelXmlPathW);
            ov::test::utils::removeFile(modelBinPathW);
            GTEST_COUT << "OK" << std::endl;
        } catch (const ov::Exception& e_next) {
            ov::test::utils::removeFile(modelXmlPathW);
            ov::test::utils::removeFile(modelBinPathW);
            ov::test::utils::removeFile(model_xml_name);
            ov::test::utils::removeFile(model_bin_name);
            FAIL() << e_next.what();
        }
    }
    ov::test::utils::removeFile(model_xml_name);
    ov::test::utils::removeFile(model_bin_name);
}
#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPOR
TEST_P(OVClassBasicTestP, smoke_registerPluginsXMLUnicodePath) {
    const std::string pluginXML = getPluginFile();

    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        GTEST_COUT << testIndex;
        std::wstring postfix = L"_" + ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring pluginsXmlW = ov::test::utils::addUnicodePostfixToPath(pluginXML, postfix);

        try {
            bool is_copy_successfully;
            is_copy_successfully = ov::test::utils::copyFile(pluginXML, pluginsXmlW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << pluginXML << "' to '"
                       << ::ov::util::wstring_to_string(pluginsXmlW) << "'";
            }

            GTEST_COUT << "Test " << testIndex << std::endl;

            ov::Core ie = createCoreWithTemplate();
            GTEST_COUT << "Core created " << testIndex << std::endl;
            OV_ASSERT_NO_THROW(ie.register_plugins(::ov::util::wstring_to_string(pluginsXmlW)));
            ov::test::utils::removeFile(pluginsXmlW);
            OV_ASSERT_NO_THROW(ie.get_versions("mock"));  // from pluginXML
            OV_ASSERT_NO_THROW(ie.get_versions(target_device));
            GTEST_COUT << "Plugin created " << testIndex << std::endl;

            OV_ASSERT_NO_THROW(ie.register_plugin(pluginName, "TEST_DEVICE"));
            OV_ASSERT_NO_THROW(ie.get_versions("TEST_DEVICE"));
            GTEST_COUT << "Plugin registered and created " << testIndex << std::endl;

            GTEST_COUT << "OK" << std::endl;
        } catch (const ov::Exception& e_next) {
            ov::test::utils::removeFile(pluginsXmlW);
            std::remove(pluginXML.c_str());
            FAIL() << e_next.what();
        }
    }
    ov::test::utils::removeFile(pluginXML);
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#endif // !OPENVINO_STATIC_LIBRARY

//
// GetVersions()
//

TEST_P(OVClassBasicTestP, getVersionsByExactDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.get_versions(target_device + ".0"));
}

TEST_P(OVClassBasicTestP, getVersionsByDeviceClassNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.get_versions(target_device));
}

TEST_P(OVClassBasicTestP, getVersionsNonEmpty) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_EQ(2, ie.get_versions(ov::test::utils::DEVICE_HETERO + std::string(":") + target_device).size());
}

//
// UnregisterPlugin
//

TEST_P(OVClassBasicTestP, unregisterExistingPluginNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    // device instance is not created yet
    ASSERT_THROW(ie.unload_plugin(target_device), ov::Exception);

    // make the first call to IE which created device instance
    ie.get_versions(target_device);
    // now, we can unregister device
    OV_ASSERT_NO_THROW(ie.unload_plugin(target_device));
}

TEST_P(OVClassBasicTestP, accessToUnregisteredPluginThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.unload_plugin(target_device), ov::Exception);
    OV_ASSERT_NO_THROW(ie.get_versions(target_device));
    OV_ASSERT_NO_THROW(ie.unload_plugin(target_device));
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::AnyMap{}));
    OV_ASSERT_NO_THROW(ie.get_versions(target_device));
    OV_ASSERT_NO_THROW(ie.unload_plugin(target_device));
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
    ASSERT_ANY_THROW(ie.get_versions(target_device));
}

TEST_P(OVClassBasicTestP, SetConfigForUnRegisteredDeviceThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.set_property("unregistered_device", {{"unsupported_key", "4"}}), ov::Exception);
}

TEST_P(OVClassBasicTestP, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::enable_profiling(true)));
}

TEST_P(OVClassBasicTestP, SetConfigAllNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(ov::enable_profiling(true)));
    OV_ASSERT_NO_THROW(ie.get_versions(target_device));
}

TEST(OVClassBasicTest, smoke_SetConfigHeteroThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(ov::test::utils::DEVICE_HETERO, ov::enable_profiling(true)));
}

TEST(OVClassBasicTest, smoke_SetConfigDevicePropertiesThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.set_property("", ov::device::properties(ov::test::utils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(ie.set_property(ov::test::utils::DEVICE_CPU,
                                 ov::device::properties(ov::test::utils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(ie.set_property(ov::test::utils::DEVICE_AUTO,
                                 ov::device::properties(ov::test::utils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(ie.set_property(ov::test::utils::DEVICE_AUTO,
                                 ov::device::properties(ov::test::utils::DEVICE_CPU, ov::num_streams(4))),
                 ov::Exception);
}

TEST_P(OVClassBasicTestP, SetConfigHeteroTargetFallbackThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities(target_device)));
}

TEST_P(OVClassBasicTestP, smoke_SetConfigHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string value;

    OV_ASSERT_NO_THROW(ie.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities(target_device)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities));
    ASSERT_EQ(target_device, value);

    OV_ASSERT_NO_THROW(ie.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities(target_device)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities));
    ASSERT_EQ(target_device, value);
}

TEST(OVClassBasicTest, smoke_SetConfigAutoNoThrows) {
    ov::Core ie = createCoreWithTemplate();

    // priority config test
    ov::hint::Priority value;
    OV_ASSERT_NO_THROW(ie.set_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::LOW)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::LOW);
    OV_ASSERT_NO_THROW(ie.set_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::MEDIUM)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::MEDIUM);
    OV_ASSERT_NO_THROW(ie.set_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::HIGH)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::HIGH);
}

TEST_P(OVClassSpecificDeviceTestSetConfig, SetConfigSpecificDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::string deviceID, cleartarget_device;
    auto pos = target_device.find('.');
    if (pos != std::string::npos) {
        cleartarget_device = target_device.substr(0, pos);
        deviceID =  target_device.substr(pos + 1,  target_device.size());
    }
    if (!supportsDeviceID(ie, cleartarget_device) || !supportsAvailableDevices(ie, cleartarget_device)) {
        GTEST_FAIL();
    }
    auto deviceIDs = ie.get_property(cleartarget_device, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL();
    }

    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::enable_profiling(true)));
    bool value = false;
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::enable_profiling));
    ASSERT_TRUE(value);
}

TEST_P(OVClassSetModelPriorityConfigTest, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    // priority config test
    ov::hint::Priority value;
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::hint::model_priority(ov::hint::Priority::LOW)));
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::LOW);
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::hint::model_priority(ov::hint::Priority::MEDIUM)));
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::MEDIUM);
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::hint::model_priority(ov::hint::Priority::HIGH)));
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::HIGH);
}

TEST_P(OVClassSetExecutionModeHintConfigTest, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    OV_ASSERT_PROPERTY_SUPPORTED(ov::hint::execution_mode);

    ov::hint::ExecutionMode defaultMode{};
    ASSERT_NO_THROW(defaultMode = ie.get_property(target_device, ov::hint::execution_mode));
    (void)defaultMode;

    ie.set_property(target_device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    ASSERT_EQ(ov::hint::ExecutionMode::ACCURACY, ie.get_property(target_device, ov::hint::execution_mode));
    ie.set_property(target_device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    ASSERT_EQ(ov::hint::ExecutionMode::PERFORMANCE, ie.get_property(target_device, ov::hint::execution_mode));
}

TEST_P(OVClassSetEnableCpuPinningHintConfigTest, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    OV_ASSERT_PROPERTY_SUPPORTED(ov::hint::enable_cpu_pinning);

    bool defaultMode{};
    ASSERT_NO_THROW(defaultMode = ie.get_property(target_device, ov::hint::enable_cpu_pinning));
    (void)defaultMode;

    ASSERT_EQ(true, ie.get_property(target_device, ov::hint::enable_cpu_pinning));

    ie.set_property(target_device, ov::hint::enable_cpu_pinning(false));
    ASSERT_EQ(false, ie.get_property(target_device, ov::hint::enable_cpu_pinning));
    ie.set_property(target_device, ov::hint::enable_cpu_pinning(true));
    ASSERT_EQ(true, ie.get_property(target_device, ov::hint::enable_cpu_pinning));
}

TEST_P(OVClassSetSchedulingCoreTypeHintConfigTest, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    OV_ASSERT_PROPERTY_SUPPORTED(ov::hint::scheduling_core_type);

    ov::hint::SchedulingCoreType defaultMode{};
    ASSERT_NO_THROW(defaultMode = ie.get_property(target_device, ov::hint::scheduling_core_type));
    (void)defaultMode;

    ASSERT_EQ(ov::hint::SchedulingCoreType::ANY_CORE, ie.get_property(target_device, ov::hint::scheduling_core_type));

    ie.set_property(target_device, ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY));
    ASSERT_EQ(ov::hint::SchedulingCoreType::PCORE_ONLY, ie.get_property(target_device, ov::hint::scheduling_core_type));
    ie.set_property(target_device, ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY));
    ASSERT_EQ(ov::hint::SchedulingCoreType::ECORE_ONLY, ie.get_property(target_device, ov::hint::scheduling_core_type));
    ie.set_property(target_device, ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ANY_CORE));
    ASSERT_EQ(ov::hint::SchedulingCoreType::ANY_CORE, ie.get_property(target_device, ov::hint::scheduling_core_type));
}

TEST_P(OVClassSetEnableHyperThreadingHintConfigTest, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    OV_ASSERT_PROPERTY_SUPPORTED(ov::hint::enable_hyper_threading);

    bool defaultMode{};
    ASSERT_NO_THROW(defaultMode = ie.get_property(target_device, ov::hint::enable_hyper_threading));
    (void)defaultMode;

    ASSERT_EQ(true, ie.get_property(target_device, ov::hint::enable_hyper_threading));

    ie.set_property(target_device, ov::hint::enable_hyper_threading(false));
    ASSERT_EQ(false, ie.get_property(target_device, ov::hint::enable_hyper_threading));
    ie.set_property(target_device, ov::hint::enable_hyper_threading(true));
    ASSERT_EQ(true, ie.get_property(target_device, ov::hint::enable_hyper_threading));
}

TEST_P(OVClassSetDevicePriorityConfigTest, SetConfigAndCheckGetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string devicePriority;
    OV_ASSERT_NO_THROW(ie.set_property(target_device, configuration));
    OV_ASSERT_NO_THROW(devicePriority = ie.get_property(target_device, ov::device::priorities));
    ASSERT_EQ(devicePriority, configuration[ov::device::priorities.name()].as<std::string>());
}

TEST(OVClassBasicTest, SetCacheDirPropertyCoreNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    // Cache_dir property test
    ov::Any value;
    OV_ASSERT_NO_THROW(ie.set_property(ov::cache_dir("./tmp_cache_dir")));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::cache_dir.name()));
    EXPECT_EQ(value.as<std::string>(), std::string("./tmp_cache_dir"));
}

TEST(OVClassBasicTest, SetTBBForceTerminatePropertyCoreNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    bool value = true;
    OV_ASSERT_NO_THROW(ie.set_property(ov::force_tbb_terminate(false)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::force_tbb_terminate.name()).as<bool>());
    EXPECT_FALSE(value);
    OV_ASSERT_NO_THROW(ie.set_property(ov::force_tbb_terminate(true)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::force_tbb_terminate.name()).as<bool>());
    EXPECT_TRUE(value);
}

TEST(OVClassBasicTest, SetEnableMmapPropertyCoreNoThrow) {
    ov::Core ie;

    bool value = true;
    OV_ASSERT_NO_THROW(ie.set_property(ov::enable_mmap(false)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::enable_mmap.name()).as<bool>());
    EXPECT_FALSE(value);
    OV_ASSERT_NO_THROW(ie.set_property(ov::enable_mmap(true)));
    OV_ASSERT_NO_THROW(value = ie.get_property(ov::enable_mmap.name()).as<bool>());
    EXPECT_TRUE(value);
}

TEST(OVClassBasicTest, GetUnsupportedPropertyCoreThrow) {
    ov::Core ie = createCoreWithTemplate();

    // Unsupported property test
    ASSERT_THROW(ie.get_property("unsupported_property"), ov::Exception);
}

TEST_P(OVClassSetLogLevelConfigTest, SetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    // log level
    ov::log::Level logValue;
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::log::level(ov::log::Level::NO)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(target_device, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::NO);
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::log::level(ov::log::Level::ERR)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(target_device, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::ERR);
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::log::level(ov::log::Level::WARNING)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(target_device, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::WARNING);
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::log::level(ov::log::Level::INFO)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(target_device, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::INFO);
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::log::level(ov::log::Level::DEBUG)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(target_device, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::DEBUG);
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::log::level(ov::log::Level::TRACE)));
    OV_ASSERT_NO_THROW(logValue = ie.get_property(target_device, ov::log::level));
    EXPECT_EQ(logValue, ov::log::Level::TRACE);
}

//
// QueryNetwork
//

TEST_P(OVClassNetworkTestP, QueryNetworkActualThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.query_model(actualNetwork, ov::test::utils::DEVICE_HETERO + std::string(":") + target_device));
}

TEST_P(OVClassNetworkTestP, QueryNetworkActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ie.query_model(actualNetwork, target_device);
}

TEST_P(OVClassNetworkTestP, QueryNetworkWithKSO) {
    ov::Core ie = createCoreWithTemplate();

    auto rl_map = ie.query_model(ksoNetwork, target_device);
    auto func = ksoNetwork;
    for (const auto& op : func->get_ops()) {
        if (!rl_map.count(op->get_friendly_name())) {
            FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
        }
    }
}

TEST_P(OVClassSeveralDevicesTestQueryNetwork, QueryNetworkActualSeveralDevicesNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
    if (!supportsAvailableDevices(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    ASSERT_LT(deviceIDs.size(), target_devices.size());

    std::string multi_target_device = ov::test::utils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : target_devices) {
        multi_target_device += dev_name;
        if (&dev_name != &(target_devices.back())) {
            multi_target_device += ",";
        }
    }
    OV_ASSERT_NO_THROW(ie.query_model(actualNetwork, multi_target_device));
}

TEST_P(OVClassNetworkTestP, SetAffinityWithConstantBranches) {
    ov::Core ie = createCoreWithTemplate();

    std::shared_ptr<ov::Model> func;
    {
        ov::PartialShape shape({1, 84});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto matMulWeights = ov::op::v0::Constant::create(ov::element::Type_t::f32, {10, 84}, {1});
        auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(matMulWeights);
        auto gConst1 = ov::op::v0::Constant::create(ov::element::Type_t::i32, {1}, {1});
        auto gConst2 = ov::op::v0::Constant::create(ov::element::Type_t::i64, {}, {0});
        auto gather = std::make_shared<ov::op::v1::Gather>(shapeOf, gConst1, gConst2);
        auto concatConst = ov::op::v0::Constant::create(ov::element::Type_t::i64, {1}, {1});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{concatConst, gather}, 0);
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(relu, concat, false);
        auto matMul = std::make_shared<ov::op::v0::MatMul>(reshape, matMulWeights, false, true);
        auto matMulBias = ov::op::v0::Constant::create(ov::element::Type_t::f32, {1, 10}, {1});
        auto addBias = std::make_shared<ov::op::v1::Add>(matMul, matMulBias);
        auto result = std::make_shared<ov::op::v0::Result>(addBias);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        func = std::make_shared<ov::Model>(results, params);
    }

    auto rl_map = ie.query_model(func, target_device);
    for (const auto& op : func->get_ops()) {
        if (!rl_map.count(op->get_friendly_name())) {
            FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
        }
    }
    for (const auto& op : func->get_ops()) {
        std::string affinity = rl_map[op->get_friendly_name()];
        op->get_rt_info()["affinity"] = affinity;
    }
    auto exeNetwork = ie.compile_model(func, target_device);
}

TEST_P(OVClassNetworkTestP, SetAffinityWithKSO) {
    ov::Core ie = createCoreWithTemplate();

    auto rl_map = ie.query_model(ksoNetwork, target_device);
    auto func = ksoNetwork;
    for (const auto& op : func->get_ops()) {
        if (!rl_map.count(op->get_friendly_name())) {
            FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
        }
    }
    for (const auto& op : func->get_ops()) {
        std::string affinity = rl_map[op->get_friendly_name()];
        op->get_rt_info()["affinity"] = affinity;
    }
    auto exeNetwork = ie.compile_model(ksoNetwork, target_device);
}

TEST_P(OVClassNetworkTestP, QueryNetworkHeteroActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ov::SupportedOpsMap res;
    OV_ASSERT_NO_THROW(
        res = ie.query_model(actualNetwork, ov::test::utils::DEVICE_HETERO, ov::device::priorities(target_device)));
    ASSERT_LT(0, res.size());
}

TEST_P(OVClassNetworkTestP, QueryNetworkMultiNoThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.query_model(actualNetwork, ov::test::utils::DEVICE_MULTI));
}

TEST(OVClassBasicTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string target_device = ov::test::utils::DEVICE_HETERO;

    std::vector<ov::PropertyName> t;
    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::supported_properties));

    std::cout << "Supported HETERO properties: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << " is_mutable: " << str.is_mutable() << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassBasicTestP, smoke_GetMetricSupportedConfigKeysHeteroThrows) {
    ov::Core ie = createCoreWithTemplate();
    // TODO: check
    std::string real_target_device = ov::test::utils::DEVICE_HETERO + std::string(":") + target_device;
    ASSERT_THROW(ie.get_property(real_target_device, ov::supported_properties), ov::Exception);
}

TEST_P(OVClassGetMetricTest_SUPPORTED_METRICS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> t;

    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::supported_properties));

    std::cout << "Supported properties: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << " is_mutable: " << str.is_mutable() << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> t;

    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::supported_properties));

    std::cout << "Supported config values: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << " is_mutable: " << str.is_mutable() << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassGetMetricTest_AVAILABLE_DEVICES, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<std::string> t;

    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::available_devices));

    std::cout << "Available devices: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::available_devices);
}

TEST_P(OVClassGetMetricTest_FULL_DEVICE_NAME, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string t;

    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::device::full_name));
    std::cout << "Full device name: " << std::endl << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::full_name);
}

TEST_P(OVClassGetMetricTest_FULL_DEVICE_NAME_with_DEVICE_ID, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string t;

    if (supportsDeviceID(ie, target_device)) {
        auto device_ids = ie.get_property(target_device, ov::available_devices);
        ASSERT_GT(device_ids.size(), 0);
        OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::device::full_name, ov::device::id(device_ids.front())));
        std::cout << "Device " << device_ids.front() << " " <<  ", Full device name: " << std::endl << t << std::endl;
        OV_ASSERT_PROPERTY_SUPPORTED(ov::device::full_name);
    } else {
        GTEST_FAIL() << "Device id is not supported";
    }
}

TEST_P(OVClassGetMetricTest_DEVICE_UUID, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ov::device::UUID t;

    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::device::uuid));
    std::cout << "Device uuid: " << std::endl << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::uuid);
}

TEST_P(OVClassGetMetricTest_DEVICE_LUID, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ov::device::LUID t;

    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::device::luid));
    std::cout << "Device luid: " << std::endl << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::luid);
}

TEST_P(OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<std::string> t;
    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::device::capabilities));
    std::cout << "Optimization capabilities: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << std::endl;
    }
    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::capabilities);
}

TEST_P(OVClassGetMetricTest_MAX_BATCH_SIZE, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    uint32_t max_batch_size = 0;

    ASSERT_NO_THROW(max_batch_size = ie.get_property(target_device, ov::max_batch_size));

    std::cout << "Max batch size: " << max_batch_size << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::max_batch_size);
}

TEST_P(OVClassGetMetricTest_DEVICE_GOPS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::cout << "Device GOPS: " << std::endl;
    for (auto&& kv : ie.get_property(target_device, ov::device::gops)) {
        std::cout << kv.first << ": " << kv.second << std::endl;
    }
    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::gops);
}

TEST_P(OVClassGetMetricTest_DEVICE_TYPE, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::type);
    ov::device::Type t = {};
    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::device::type));
    std::cout << "Device Type: " << t << std::endl;
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    unsigned int start{0}, end{0}, step{0};

    ASSERT_NO_THROW(std::tie(start, end, step) = ie.get_property(target_device, ov::range_for_async_infer_requests));

    std::cout << "Range for async infer requests: " << std::endl
    << start << std::endl
    << end << std::endl
    << step << std::endl
    << std::endl;

    ASSERT_LE(start, end);
    ASSERT_GE(step, 1u);
    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_async_infer_requests);
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_STREAMS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    unsigned int start = 0, end = 0;

    ASSERT_NO_THROW(std::tie(start, end) = ie.get_property(target_device, ov::range_for_streams));

    std::cout << "Range for streams: " << std::endl
    << start << std::endl
    << end << std::endl
    << std::endl;

    ASSERT_LE(start, end);
    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_streams);
}

TEST_P(OVClassGetMetricTest_ThrowUnsupported, GetMetricThrow) {
    ov::Core ie = createCoreWithTemplate();

    ASSERT_THROW(ie.get_property(target_device, "unsupported_metric"), ov::Exception);
}

TEST_P(OVClassGetConfigTest, GetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> configValues;

    OV_ASSERT_NO_THROW(configValues = ie.get_property(target_device, ov::supported_properties));

    for (auto&& confKey : configValues) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassGetConfigTest, GetConfigHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> configValues;
    OV_ASSERT_NO_THROW(configValues = ie.get_property(target_device, ov::supported_properties));

    for (auto&& confKey : configValues) {
        OV_ASSERT_NO_THROW(ie.get_property(target_device, confKey));
    }
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroThrow) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.get_property(ov::test::utils::DEVICE_HETERO, "unsupported_config"), ov::Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroWithDeviceThrow) {
    ov::Core ie = createCoreWithTemplate();

    ASSERT_THROW(ie.get_property(ov::test::utils::DEVICE_HETERO + std::string(":") + target_device,
                                   ov::device::priorities),
                 ov::Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigThrow) {
    ov::Core ie = createCoreWithTemplate();

    ASSERT_THROW(ie.get_property(target_device, "unsupported_config"), ov::Exception);
}

TEST_P(OVClassSpecificDeviceTestGetConfig, GetConfigSpecificDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ov::Any p;

    std::string deviceID, clear_target_device;
    auto pos = target_device.find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_device.substr(0, pos);
        deviceID =  target_device.substr(pos + 1,  target_device.size());
    }
    if (!supportsDeviceID(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
    if (!supportsAvailableDevices(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL() << "No DeviceID" << std::endl;
    }

    std::vector<ov::PropertyName> configValues;
    OV_ASSERT_NO_THROW(configValues = ie.get_property(target_device, ov::supported_properties));

    for (auto &&confKey : configValues) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, confKey));
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
TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        ie.query_model(actualNetwork,
                       ov::test::utils::DEVICE_HETERO,
                       ov::device::priorities(target_device + "." + deviceIDs[0], target_device));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithDeviceID) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        ie.query_model(simpleNetwork, target_device + "." + deviceIDs[0]);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.query_model(actualNetwork, target_device + ".110"), ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithInvalidDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.query_model(actualNetwork, target_device + ".l0"), ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.query_model(actualNetwork,
                                    ov::test::utils::DEVICE_HETERO,
                                    ov::device::priorities(target_device + ".100", target_device)),
                     ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

using OVClassNetworkTestP = OVClassBaseTestP;

//
// LoadNetwork
//

TEST_P(OVClassNetworkTestP, LoadNetworkActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, target_device));
}

TEST_P(OVClassNetworkTestP, LoadNetworkMultiWithoutSettingDevicePrioritiesThrows) {
    ov::Core ie = createCoreWithTemplate();
    try {
        ie.compile_model(actualNetwork, ov::test::utils::DEVICE_MULTI);
    } catch (ov::Exception& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("KEY_MULTI_DEVICE_PRIORITIES key is not set for"),
                            error.what());
    } catch (...) {
        FAIL() << "compile_model is failed for unexpected reason.";
    }
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, ov::test::utils::DEVICE_HETERO + std::string(":") + target_device));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, ov::test::utils::DEVICE_HETERO, ov::device::priorities(target_device)));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceUsingDevicePropertiesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork,
        ov::test::utils::DEVICE_HETERO,
        ov::device::priorities(target_device),
        ov::device::properties(target_device,
            ov::enable_profiling(true))));
}

TEST_P(OVClassNetworkTestP, LoadNetworkCreateDefaultExecGraphResult) {
    auto ie = createCoreWithTemplate();
    auto net = ie.compile_model(actualNetwork, target_device);
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

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsAvailableDevices(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect DeviceID" << std::endl;

    std::string multitarget_device = ov::test::utils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : target_devices) {
        multitarget_device += dev_name;
        if (&dev_name != &(target_devices.back())) {
            multitarget_device += ",";
        }
    }
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, multitarget_device));
}

//
// LoadNetwork with HETERO on particular device
//
TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        std::string heteroDevice =
                ov::test::utils::DEVICE_HETERO + std::string(":") + target_device + "." + deviceIDs[0] + "," + target_device;
        OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, heteroDevice));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        OV_ASSERT_NO_THROW(ie.compile_model(simpleNetwork, target_device + "." + deviceIDs[0]));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, target_device + ".10"), ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkWithCondidateDeviceListContainedMetaPluginTest, LoadNetworkRepeatedlyWithMetaPluginTestThrow) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.compile_model(actualNetwork, target_device, configuration), ov::Exception);
}

TEST_P(OVClassLoadNetworkWithCorrectPropertiesTest, LoadNetworkWithCorrectPropertiesTest) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, target_device, configuration));
}

TEST_P(OVClassLoadNetworkAndCheckSecondaryPropertiesTest, LoadNetworkAndCheckSecondaryPropertiesTest) {
    ov::Core ie = createCoreWithTemplate();
    ov::CompiledModel model;
    OV_ASSERT_NO_THROW(model = ie.compile_model(actualNetwork, target_device, configuration));
    ov::AnyMap property = configuration;
    ov::AnyMap::iterator it = configuration.end();
    // device properties in form ov::device::properties(DEVICE, ...) has the first priority
    for (it = configuration.begin(); it != configuration.end(); it++) {
        if ((it->first.find(ov::device::properties.name()) != std::string::npos) &&
            (it->first != ov::device::properties.name())) {
            break;
        }
    }
    if (it != configuration.end()) {
        // DEVICE_PROPERTIES_<DEVICE_NAME> found
        property = it->second.as<ov::AnyMap>();
    } else {
        // search for DEVICE_PROPERTIES
        it = configuration.find(ov::device::properties.name());
        ASSERT_TRUE(it != configuration.end());
        property = it->second.as<ov::AnyMap>().begin()->second.as<ov::AnyMap>();
        if (it == configuration.end()) {
            it = configuration.find(ov::num_streams.name());
        }
    }
    ASSERT_TRUE(property.count(ov::num_streams.name()));
    auto actual = property.at(ov::num_streams.name()).as<int32_t>();
    ov::Any value;
    //AutoExcutableNetwork GetMetric() does not support key ov::num_streams
    OV_ASSERT_NO_THROW(value = model.get_property(ov::num_streams.name()));
    int32_t expect = value.as<int32_t>();
    ASSERT_EQ(actual, expect);
}

TEST_P(OVClassLoadNetWorkReturnDefaultHintTest, LoadNetworkReturnDefaultHintTest) {
    ov::Core ie = createCoreWithTemplate();
    ov::CompiledModel model;
    ov::hint::PerformanceMode value;
    OV_ASSERT_NO_THROW(model = ie.compile_model(actualNetwork, target_device, configuration));
    OV_ASSERT_NO_THROW(value = model.get_property(ov::hint::performance_mode));
    if (target_device.find("AUTO") != std::string::npos) {
        ASSERT_EQ(value, ov::hint::PerformanceMode::LATENCY);
    } else {
        ASSERT_EQ(value, ov::hint::PerformanceMode::THROUGHPUT);
    }
}

TEST_P(OVClassLoadNetWorkDoNotReturnDefaultHintTest, LoadNetworkDoNotReturnDefaultHintTest) {
    ov::Core ie = createCoreWithTemplate();
    ov::CompiledModel model;
    ov::hint::PerformanceMode value;
    OV_ASSERT_NO_THROW(model = ie.compile_model(actualNetwork, target_device, configuration));
    OV_ASSERT_NO_THROW(value = model.get_property(ov::hint::performance_mode));
    if (target_device.find("AUTO") != std::string::npos) {
        ASSERT_NE(value, ov::hint::PerformanceMode::LATENCY);
    } else {
        ASSERT_EQ(value, ov::hint::PerformanceMode::THROUGHPUT);
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithInvalidDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, target_device + ".l0"), ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                      "HETERO",
                                       ov::device::priorities(target_device + ".100", ov::test::utils::DEVICE_CPU)),
                     ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROAndDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                      ov::test::utils::DEVICE_HETERO,
                                      ov::device::priorities(target_device, ov::test::utils::DEVICE_CPU),
                                      ov::device::id("110")),
                     ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

//
// LoadNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROwithMULTINoThrow) {
    ov::Core ie = createCoreWithTemplate();
    if (supportsDeviceID(ie, target_device) && supportsAvailableDevices(ie, target_device)) {
        std::string devices;
        auto availableDevices = ie.get_property(target_device, ov::available_devices);
        for (auto&& device : availableDevices) {
            devices += target_device + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        ie.compile_model(actualNetwork,
                            ov::test::utils::DEVICE_HETERO,
                            ov::device::properties(ov::test::utils::DEVICE_MULTI,
                                            ov::device::priorities(devices)),
                            ov::device::properties(ov::test::utils::DEVICE_HETERO,
                                            ov::device::priorities(ov::test::utils::DEVICE_MULTI, target_device)));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkMULTIwithHETERONoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device) && supportsAvailableDevices(ie, target_device)) {
        std::string hetero_devices;
        auto device_ids = ie.get_property(target_device, ov::available_devices);
        for (auto&& device_id : device_ids) {
            hetero_devices += target_device + std::string(".") + device_id;
            if (&device_id != &device_ids.back())
                hetero_devices += ',';
        }
        OV_ASSERT_NO_THROW(ie.compile_model(
            actualNetwork,
            ov::test::utils::DEVICE_MULTI,
            ov::device::properties(ov::test::utils::DEVICE_MULTI, ov::device::priorities(ov::test::utils::DEVICE_HETERO)),
            ov::device::properties(ov::test::utils::DEVICE_HETERO, ov::device::priorities(hetero_devices))));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

//
// QueryNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, QueryNetworkHETEROWithMULTINoThrow_V10) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device) && supportsAvailableDevices(ie, target_device)) {
        std::string devices;
        auto availableDevices = ie.get_property(target_device, ov::available_devices);
        for (auto&& device : availableDevices) {
            devices += target_device + '.' + device;
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
        std::string hetero_device_priorities(ov::test::utils::DEVICE_MULTI + std::string(",") + target_device);
        OV_ASSERT_NO_THROW(result = ie.query_model(
                            multinputNetwork,
                            ov::test::utils::DEVICE_HETERO,
                            ov::device::properties(ov::test::utils::DEVICE_MULTI,
                                                   ov::device::priorities(devices)),
                            ov::device::properties(ov::test::utils::DEVICE_HETERO,
                                                   ov::device::priorities(ov::test::utils::DEVICE_MULTI,
                                                                          target_device))));

        std::unordered_set<std::string> actualLayers;
        for (auto&& layer : result) {
            actualLayers.emplace(layer.first);
        }
        ASSERT_EQ(expectedLayers, actualLayers);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, QueryNetworkMULTIWithHETERONoThrow_V10) {
    ov::Core ie = createCoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
    if (!supportsAvailableDevices(ie, target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
    }
    std::string devices;
    auto device_ids = ie.get_property(target_device, ov::available_devices);
    for (auto&& device_id : device_ids) {
        devices += target_device + "." + device_id;
        if (&device_id != &(device_ids.back())) {
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
                                                ov::test::utils::DEVICE_MULTI,
                                                ov::device::properties(ov::test::utils::DEVICE_MULTI,
                                                                ov::device::priorities(ov::test::utils::DEVICE_HETERO)),
                                                ov::device::properties(ov::test::utils::DEVICE_HETERO,
                                                                ov::device::priorities(devices))));

    std::unordered_set<std::string> actualLayers;
    for (auto&& layer : result) {
        actualLayers.emplace(layer.first);
    }
    ASSERT_EQ(expectedLayers, actualLayers);
}

TEST_P(OVClassSetDefaultDeviceIDTest, SetDefaultDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto deviceIDs = ie.get_property(target_device, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL();
    }
    std::string value;
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::device::id(deviceID), ov::enable_profiling(true)));
    ASSERT_TRUE(ie.get_property(target_device, ov::enable_profiling));
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::enable_profiling.name()).as<std::string>());
    ASSERT_EQ(value, "YES");
}

TEST_P(OVClassSetGlobalConfigTest, SetGlobalConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto deviceIDs = ie.get_property(target_device, ov::available_devices);
    ov::Any ref, src;
    for (auto& dev_id : deviceIDs) {
        OV_ASSERT_NO_THROW(ie.set_property(target_device + "." + dev_id, ov::enable_profiling(false)));
    }
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::enable_profiling(true)));
    OV_ASSERT_NO_THROW(ref = ie.get_property(target_device, ov::enable_profiling.name()));

    for (auto& dev_id : deviceIDs) {
        OV_ASSERT_NO_THROW(src = ie.get_property(target_device + "." + dev_id, ov::enable_profiling.name()));
        ASSERT_EQ(src, ref);
    }
}

TEST_P(OVClassSeveralDevicesTestDefaultCore, DefaultCoreSeveralDevicesNoThrow) {
    ov::Core ie;

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
    if (!supportsAvailableDevices(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect Device ID" << std::endl;

    for (size_t i = 0; i < target_devices.size(); ++i) {
        OV_ASSERT_NO_THROW(ie.set_property(target_devices[i], ov::enable_profiling(true)));
    }
    bool res;
    for (size_t i = 0; i < target_devices.size(); ++i) {
        OV_ASSERT_NO_THROW(res = ie.get_property(target_devices[i], ov::enable_profiling));
        ASSERT_TRUE(res);
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include <openvino/runtime/properties.hpp>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/util/file_util.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>
#endif

namespace ov {
namespace test {
namespace behavior {

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
        pluginName += IE_BUILD_POSTFIX;
        if (pluginName == (std::string("openvino_template_plugin") + IE_BUILD_POSTFIX)) {
            pluginName = ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(), pluginName);
        }
    }
};

using OVClassImportExportTestP = OVClassBaseTestP;
using OVClassSetTBBForceTerminatePropertyTest = OVClassBaseTestP;


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
    std::string filename = CommonTestUtils::generateTestFilePrefix() + "_" + filePostfix;
    std::ostringstream stream;
    stream << "<ie><plugins><plugin name=\"mock\" location=\"";
    stream << ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
        std::string("mock_engine") + IE_BUILD_POSTFIX);
    stream << "\"></plugin></plugins></ie>";
    CommonTestUtils::createFile(filename, stream.str());
    return filename;
}

TEST(OVClassBasicTest, smoke_createMockEngineConfigNoThrows) {
    const std::string filename = getPluginFile();
    OV_ASSERT_NO_THROW(ov::Core ie(filename));
    CommonTestUtils::removeFile(filename.c_str());
}

TEST(OVClassBasicTest, smoke_createMockEngineConfigThrows) {
    std::string filename = CommonTestUtils::generateTestFilePrefix() + "_mock_engine.xml";
    std::string content{"<ie><plugins><plugin location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    CommonTestUtils::createFile(filename, content);
    ASSERT_THROW(ov::Core ie(filename), ov::Exception);
    CommonTestUtils::removeFile(filename.c_str());
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPOR
TEST_P(OVClassBasicTestP, smoke_registerPluginsXMLUnicodePath) {
    const std::string pluginXML = getPluginFile();

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
            OV_ASSERT_NO_THROW(ie.get_versions("mock"));  // from pluginXML
            OV_ASSERT_NO_THROW(ie.get_versions(target_device));
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
    OV_ASSERT_NO_THROW(ie.get_versions(target_device + ".0"));
}

TEST_P(OVClassBasicTestP, getVersionsByDeviceClassNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.get_versions(target_device));
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

}  // namespace behavior
}  // namespace test
}  // namespace ov

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iplugin.hpp"

using namespace ::testing;
using namespace std;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>

#    include "openvino/pass/manager.hpp"
#endif

#ifndef OPENVINO_STATIC_LIBRARY

namespace {
void mockPlugin(ov::Core& core, std::shared_ptr<ov::IPlugin>& plugin, std::shared_ptr<void>& m_so) {
    std::string libraryPath = ov::test::utils::get_mock_engine_path();
    if (!m_so)
        m_so = ov::util::load_shared_object(libraryPath.c_str());
    std::function<void(ov::IPlugin*)> injectProxyEngine =
        ov::test::utils::make_std_function<void(ov::IPlugin*)>(m_so, "InjectPlugin");

    injectProxyEngine(plugin.get());
}

void clearMockPlugin(const std::shared_ptr<void>& m_so) {
    ASSERT_TRUE(m_so);
    ov::test::utils::make_std_function<void()>(m_so, "ClearTargets")();
}
}  // namespace

TEST(RegisterPluginTests, getVersionforRegisteredPluginThrows) {
    ov::Core core;
    auto plugin = std::make_shared<ov::test::utils::MockPlugin>();
    std::shared_ptr<ov::IPlugin> base_plugin = plugin;
    std::shared_ptr<void> m_so;
    mockPlugin(core, base_plugin, m_so);
    std::string mock_plugin_name{"MOCK_REGISTERED_HARDWARE"};
    // Registered plugin with invalid so here
    OV_ASSERT_NO_THROW(core.register_plugin(
        ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                           std::string("mock_registered_engine") + OV_BUILD_POSTFIX),
        mock_plugin_name));
    ASSERT_THROW(core.get_versions("MOCK_REGISTERED_HARDWARE"), ov::Exception);
    clearMockPlugin(m_so);
}

TEST(RegisterPluginTests, getVersionforNoRegisteredPluginNoThrows) {
    ov::Core core;
    std::map<std::string, ov::Version> versions;
    OV_ASSERT_NO_THROW(versions = core.get_versions("unkown_device"));
    ASSERT_TRUE(versions.empty());

    auto plugin = std::make_shared<NiceMock<ov::MockIPlugin>>();

    ON_CALL(*plugin.get(), get_property(ov::supported_properties.name(), _))
        .WillByDefault(Return(std::vector<ov::PropertyName>{}));

    ON_CALL(*plugin.get(), get_property(ov::internal::supported_properties.name(), _))
        .WillByDefault(Return(std::vector<ov::PropertyName>{}));

    ON_CALL(*plugin.get(), set_property(_)).WillByDefault(Return());

    std::shared_ptr<ov::IPlugin> base_plugin = plugin;
    std::shared_ptr<void> m_so;
    mockPlugin(core, base_plugin, m_so);

    std::string mock_plugin_name{"MOCK_HARDWARE"};

    OV_ASSERT_NO_THROW(
        core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                std::string("mock_engine") + OV_BUILD_POSTFIX),
                             mock_plugin_name));
    OV_ASSERT_NO_THROW(core.get_versions("MOCK_HARDWARE"));
}

TEST(RegisterPluginTests, registerNewPluginNoThrows) {
    ov::Core core;
    auto plugin = std::make_shared<ov::test::utils::MockPlugin>();
    std::shared_ptr<ov::IPlugin> base_plugin = plugin;
    std::shared_ptr<void> m_so;
    mockPlugin(core, base_plugin, m_so);

    std::string mock_plugin_name{"MOCK_HARDWARE"};
    OV_ASSERT_NO_THROW(
        core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                std::string("mock_engine") + OV_BUILD_POSTFIX),
                             mock_plugin_name));
    OV_ASSERT_NO_THROW(core.get_property(mock_plugin_name, ov::supported_properties));

    core.unload_plugin(mock_plugin_name);
}

TEST(RegisterPluginTests, registerExistingPluginThrows) {
    ov::Core core;
    auto plugin = std::make_shared<ov::test::utils::MockPlugin>();
    std::shared_ptr<ov::IPlugin> base_plugin = plugin;
    std::shared_ptr<void> m_so;
    mockPlugin(core, base_plugin, m_so);

    std::string mock_plugin_name{"MOCK_HARDWARE"};
    OV_ASSERT_NO_THROW(
        core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                std::string("mock_engine") + OV_BUILD_POSTFIX),
                             mock_plugin_name));
    ASSERT_THROW(core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                         std::string("mock_engine") + OV_BUILD_POSTFIX),
                                      mock_plugin_name),
                 ov::Exception);
    clearMockPlugin(m_so);
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

TEST(RegisterPluginTests, smoke_createMockEngineConfigNoThrows) {
    const std::string filename = getPluginFile();
    OV_ASSERT_NO_THROW(ov::Core core(filename));
    ov::test::utils::removeFile(filename.c_str());
}

TEST(RegisterPluginTests, createMockEngineConfigThrows) {
    std::string filename = ov::test::utils::generateTestFilePrefix() + "_mock_engine.xml";
    std::string content{"<ie><plugins><plugin location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    ov::test::utils::createFile(filename, content);
    ASSERT_THROW(ov::Core core(filename), ov::Exception);
    ov::test::utils::removeFile(filename.c_str());
}

TEST(RegisterPluginTests2, createNonExistingConfigThrows) {
    ASSERT_THROW(ov::Core core("nonExistPlugins.xml"), ov::Exception);
}

TEST(RegisterPluginTests2, registerNonExistingPluginFileThrows) {
    ov::Core core;
    ASSERT_THROW(core.register_plugins("nonExistPlugins.xml"), ov::Exception);
}

TEST(RegisterPluginTests, unregisterNonExistingPluginThrows) {
    ov::Core core;
    ASSERT_THROW(core.unload_plugin("unkown_device"), ov::Exception);
}

TEST(RegisterPluginTests, unregisterExistingPluginNoThrow) {
    ov::Core core;

    // get registered devices
    std::vector<std::string> devices = core.get_available_devices();

    for (auto&& deviceWithID : devices) {
        // get_available_devices add to registered device DeviceID, we should remove it
        std::string device = deviceWithID.substr(0, deviceWithID.find("."));
        // now, we can unregister device
        ASSERT_NO_THROW(core.unload_plugin(device)) << "upload plugin fails: " << device;
    }
}

TEST(RegisterPluginTests, accessToUnregisteredPluginThrows) {
    ov::Core core;
    std::vector<std::string> devices = core.get_available_devices();

    for (auto&& device : devices) {
        OV_ASSERT_NO_THROW(core.get_versions(device));
        OV_ASSERT_NO_THROW(core.unload_plugin(device));
        OV_ASSERT_NO_THROW(core.set_property(device, ov::AnyMap{}));
        OV_ASSERT_NO_THROW(core.get_versions(device));
        OV_ASSERT_NO_THROW(core.unload_plugin(device));
    }
}

#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
TEST(RegisterPluginTests, registerPluginsXMLUnicodePath) {
    const std::string pluginXML = getPluginFile();

    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        GTEST_COUT << testIndex;
        std::wstring postfix = L"_" + ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring pluginsXmlW = ov::test::utils::addUnicodePostfixToPath(pluginXML, postfix);

        try {
            bool is_copy_successfully;
            is_copy_successfully = ov::test::utils::copyFile(pluginXML, pluginsXmlW);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << pluginXML << "' to '" << ::ov::util::wstring_to_string(pluginsXmlW)
                       << "'";
            }

            GTEST_COUT << "Test " << testIndex << std::endl;

            ov::Core core;

            GTEST_COUT << "Core created " << testIndex << std::endl;
            OV_ASSERT_NO_THROW(core.register_plugins(::ov::util::wstring_to_string(pluginsXmlW)));
            ov::test::utils::removeFile(pluginsXmlW);
            OV_ASSERT_NO_THROW(core.get_versions("mock"));  // from pluginXML

            std::vector<std::string> devices = core.get_available_devices();
            OV_ASSERT_NO_THROW(core.get_versions(devices.at(0)));
            GTEST_COUT << "Plugin created " << testIndex << std::endl;

            GTEST_COUT << "OK" << std::endl;
        } catch (const ov::Exception& e_next) {
            ov::test::utils::removeFile(pluginsXmlW);
            std::remove(pluginXML.c_str());
            FAIL() << e_next.what();
        }
    }
    ov::test::utils::removeFile(pluginXML);
}

#    endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#endif      // !OPENVINO_STATIC_LIBRARY

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "dev/core_impl.hpp"
#include "file_utils.h"
#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::util;

TEST(CoreTests, Throw_on_register_plugin_twice) {
    ov::Core core;
    core.register_plugin("test_plugin", "TEST_DEVICE");
    OV_EXPECT_THROW(core.register_plugin("test_plugin", "TEST_DEVICE"),
                    ov::Exception,
                    ::testing::HasSubstr("Device with \"TEST_DEVICE\"  is already registered in the OpenVINO Runtime"));
}

TEST(CoreTests, Throw_on_register_plugins_twice) {
    ov::Core core;

    auto get_plugin_xml = [&]() -> std::string {
        std::string plugins_xml = "test_plugins.xml";
        std::ofstream file(plugins_xml);
        file << "<ie><plugins><plugin location=\"libtest_plugin.so\" name=\"TEST_DEVICE\"></plugin></plugins></ie>";
        file.flush();
        file.close();
        return plugins_xml;
    };

    core.register_plugins(get_plugin_xml());
    OV_EXPECT_THROW(core.register_plugins(get_plugin_xml()),
                    ov::Exception,
                    ::testing::HasSubstr("Device with \"TEST_DEVICE\"  is already registered in the OpenVINO Runtime"));
}

TEST(CoreTests_get_plugin_path_from_xml, Use_abs_path_as_is) {
    auto xml_path = "path_to_plugins.xml";
    auto lib_path = ov::util::get_absolute_file_path("test_name.ext");  // CWD/test_name.ext
    for (auto as_abs_only : std::vector<bool>{true, false}) {
        auto abs_path = from_file_path(ov::get_plugin_path(lib_path, xml_path, as_abs_only));
        EXPECT_TRUE(is_absolute_file_path(abs_path));
        EXPECT_STREQ(abs_path.c_str(), lib_path.c_str());
    }
}

TEST(CoreTests_get_plugin_path_from_xml, Convert_relative_path_as_relative_to_xmldir) {
    auto xml_path = "path_to_plugins.xml";
    auto lib_path = FileUtils::makePath(std::string("."), std::string("test_name.ext"));  // ./test_name.ext
    for (auto as_abs_only : std::vector<bool>{true, false}) {
        auto abs_path = from_file_path(ov::get_plugin_path(lib_path, xml_path, as_abs_only));  // XMLDIR/test_name.ext
        EXPECT_TRUE(is_absolute_file_path(abs_path));

        auto ref_path = ov::util::get_absolute_file_path(lib_path);
        EXPECT_STREQ(abs_path.c_str(), ref_path.c_str());  // XMLDIR/test_name.ext == CWD/test_name.ext
    }
}

TEST(CoreTests_get_plugin_path_from_xml, Convert_filename_to_abs_path_if_as_abs_only) {
    auto xml_path = "path_to_plugins.xml";
    auto name = "test_name.ext";                                                // test_name.ext
    auto abs_path = from_file_path(ov::get_plugin_path(name, xml_path, true));  // XMLDIR/libtest_name.ext.so
    EXPECT_TRUE(is_absolute_file_path(abs_path));

    auto lib_name = FileUtils::makePluginLibraryName({}, std::string(name));
    auto ref_path = ov::util::get_absolute_file_path(lib_name);
    EXPECT_STREQ(abs_path.c_str(), ref_path.c_str());  // XMLDIR/libtest_name.ext.so == CWD/libtest_name.ext.so
}

TEST(CoreTests_get_plugin_path_from_xml, Use_filename_if_not_as_abs_only) {
    auto xml_path = "path_to_plugins.xml";
    auto name = "test_name.ext";                                          // test_name.ext
    auto lib_name = from_file_path(ov::get_plugin_path(name, xml_path));  // libtest_name.ext.so
    auto ref_name = FileUtils::makePluginLibraryName({}, std::string(name));
    EXPECT_STREQ(lib_name.c_str(), ref_name.c_str());
}

TEST(CoreTests_get_plugin_path, Use_abs_path_as_is) {
    auto lib_name = FileUtils::makePluginLibraryName({}, std::string("test_name"));  // libtest_name.so
    auto lib_path = ov::util::get_absolute_file_path(lib_name);
    auto abs_path = from_file_path(ov::get_plugin_path(lib_path));
    EXPECT_TRUE(is_absolute_file_path(abs_path));
    EXPECT_STREQ(abs_path.c_str(), lib_path.c_str());
}

TEST(CoreTests_get_plugin_path, Relative_path_is_from_workdir) {
    auto lib_name = FileUtils::makePluginLibraryName(std::string("."), std::string("test_name"));  // ./libtest_name.so
    auto abs_path = from_file_path(ov::get_plugin_path(lib_name));
    EXPECT_TRUE(is_absolute_file_path(abs_path));
    EXPECT_STREQ(abs_path.c_str(), get_absolute_file_path(lib_name).c_str());
}

class CoreTests_get_plugin_path_Class : public ::testing::Test {
public:
    void SetUp() override {
        std::ofstream file(lib_path);
        file << "not empty";
        file.flush();
        file.close();
    }

    void TearDown() override {
        std::remove(lib_path.c_str());
    }

    std::string lib_name = FileUtils::makePluginLibraryName({}, std::string("test_name"));  // libtest_name.so
    std::string lib_path = ov::util::get_absolute_file_path(lib_name);                      // CWD/libtest_name.so
};

TEST_F(CoreTests_get_plugin_path_Class, Filename_is_from_workdir_if_exists) {
    auto abs_path = from_file_path(ov::get_plugin_path(lib_name));  // libtest_name.so -> CWD/libtest_name.so
    EXPECT_TRUE(is_absolute_file_path(abs_path));
    EXPECT_STREQ(abs_path.c_str(), get_absolute_file_path(lib_name).c_str());
}

TEST(CoreTests_get_plugin_path, Use_filename_as_is_if_not_exist_in_workdir) {
    auto lib_name = "test_name.ext";
    auto abs_path = from_file_path(ov::get_plugin_path(lib_name));  // libtest_name.ext.so -> libtest_name.ext.so
    EXPECT_FALSE(is_absolute_file_path(abs_path));

    auto ref_path = FileUtils::makePluginLibraryName({}, std::string(lib_name));
    EXPECT_STREQ(abs_path.c_str(), ref_path.c_str());
}

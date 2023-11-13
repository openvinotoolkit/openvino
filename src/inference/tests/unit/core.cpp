// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include <fstream>

#include "common_test_utils/test_assertions.hpp"
#include "file_utils.h"
#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::util;

TEST(CoreTests, ThrowOnRegisterPluginTwice) {
    ov::Core core;
    core.register_plugin("test_plugin", "TEST_DEVICE");
    OV_EXPECT_THROW(core.register_plugin("test_plugin", "TEST_DEVICE"),
                    ov::Exception,
                    ::testing::HasSubstr("Device with \"TEST_DEVICE\"  is already registered in the OpenVINO Runtime"));
}

TEST(CoreTests, NoThrowOnRegisterPluginsTwice) {
    ov::Core core;

    auto getPluginXml = [&]() -> std::string {
        std::string pluginsXML = "test_plugins.xml";
        std::ofstream file(pluginsXML);
        file << "<ie><plugins><plugin location=\"libtest_plugin.so\" name=\"TEST_DEVICE\"></plugin></plugins></ie>";
        file.flush();
        file.close();
        return pluginsXML;
    };

    core.register_plugins(getPluginXml());
    EXPECT_NO_THROW(core.register_plugins(getPluginXml()));
}

TEST(CoreTests_getPluginPath_FromXML, UseAbsPathAsIs) {
    auto xmlPath = "path_to_plugins.xml";
    auto libPath = ov::util::get_absolute_file_path("test_name.ext");  // CWD/test_name.ext
    for (auto asAbsOnly : std::vector<bool>{true, false}) {
        auto absPath = from_file_path(get_plugin_path(libPath, xmlPath, asAbsOnly));
        EXPECT_TRUE(is_absolute_file_path(absPath));
        EXPECT_STREQ(absPath.c_str(), libPath.c_str());
    }
}

TEST(CoreTests_getPluginPath_FromXML, ConvertRelativePathAsRelativeToXMLDir) {
    auto xmlPath = "path_to_plugins.xml";
    auto libPath = FileUtils::makePath(std::string("."), std::string("test_name.ext"));  // ./test_name.ext
    for (auto asAbsOnly : std::vector<bool>{true, false}) {
        auto absPath = from_file_path(get_plugin_path(libPath, xmlPath, asAbsOnly));  // XMLDIR/test_name.ext
        EXPECT_TRUE(is_absolute_file_path(absPath));

        auto refPath = ov::util::get_absolute_file_path(libPath);
        EXPECT_STREQ(absPath.c_str(), refPath.c_str());  // XMLDIR/test_name.ext == CWD/test_name.ext
    }
}

TEST(CoreTests_getPluginPath_FromXML, ConvertFileNameToAbsPathIfAsAbsOnly) {
    auto xmlPath = "path_to_plugins.xml";
    auto name = "test_name.ext";                                            // test_name.ext
    auto absPath = from_file_path(get_plugin_path(name, xmlPath, true));  // XMLDIR/libtest_name.ext.so
    EXPECT_TRUE(is_absolute_file_path(absPath));

    auto libName = FileUtils::makePluginLibraryName({}, std::string(name));
    auto refPath = ov::util::get_absolute_file_path(libName);
    EXPECT_STREQ(absPath.c_str(), refPath.c_str());  // XMLDIR/libtest_name.ext.so == CWD/libtest_name.ext.so
}

TEST(CoreTests_getPluginPath_FromXML, UseFileNameIfNotAsAbsOnly) {
    auto xmlPath = "path_to_plugins.xml";
    auto name = "test_name.ext";                                      // test_name.ext
    auto libName = from_file_path(get_plugin_path(name, xmlPath));  // libtest_name.ext.so
    auto refName = FileUtils::makePluginLibraryName({}, std::string(name));
    EXPECT_STREQ(libName.c_str(), refName.c_str());
}

TEST(CoreTests_getPluginPath, UseAbsPathAsIs) {
    auto libName = FileUtils::makePluginLibraryName({}, std::string("test_name"));  // libtest_name.so
    auto libPath = ov::util::get_absolute_file_path(libName);
    auto absPath = from_file_path(get_plugin_path(libPath));
    EXPECT_TRUE(is_absolute_file_path(absPath));
    EXPECT_STREQ(absPath.c_str(), libPath.c_str());
}

TEST(CoreTests_getPluginPath, RelativePathIsFromWorkDir) {
    auto libName = FileUtils::makePluginLibraryName(std::string("."), std::string("test_name"));  // ./libtest_name.so
    auto absPath = from_file_path(get_plugin_path(libName));
    EXPECT_TRUE(is_absolute_file_path(absPath));
    EXPECT_STREQ(absPath.c_str(), get_absolute_file_path(libName).c_str());
}

class CoreTests_getPluginPath_Class : public ::testing::Test {
public:
    void SetUp() override {
        std::ofstream file(libPath);
        file << "not empty";
        file.flush();
        file.close();
    }

    void TearDown() override {
        std::remove(libPath.c_str());
    }

    std::string libName = FileUtils::makePluginLibraryName({}, std::string("test_name"));  // libtest_name.so
    std::string libPath = ov::util::get_absolute_file_path(libName);                       // CWD/libtest_name.so
};

TEST_F(CoreTests_getPluginPath_Class, FileNameIsFromWorkDirIfExists) {
    auto absPath = from_file_path(get_plugin_path(libName));  // libtest_name.so -> CWD/libtest_name.so
    EXPECT_TRUE(is_absolute_file_path(absPath));
    EXPECT_STREQ(absPath.c_str(), get_absolute_file_path(libName).c_str());
}

TEST(CoreTests_getPluginPath, UseFileNameAsIsIfNotExistInWorkDir) {
    auto libName = "test_name.ext";
    auto absPath = from_file_path(get_plugin_path(libName));  // libtest_name.ext.so -> libtest_name.ext.so
    EXPECT_FALSE(is_absolute_file_path(absPath));

    auto refPath = FileUtils::makePluginLibraryName({}, std::string(libName));
    EXPECT_STREQ(absPath.c_str(), refPath.c_str());
}
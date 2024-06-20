// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

#ifndef OPENVINO_STATIC_LIBRARY

static void create_plugin_xml(const std::string& file_name, const std::string& plugin_name = "1") {
    std::ofstream file(file_name);

    file << "<ie><plugins><plugin location=\"";
    file << ov::test::utils::getExecutableDirectory();
    file << ov::util::FileTraits<char>::file_separator;
    file << ov::util::FileTraits<char>::library_prefix();
    file << "mock_engine";
    file << OV_BUILD_POSTFIX;
    file << ov::util::FileTraits<char>::dot_symbol;
    file << ov::util::FileTraits<char>::library_ext();
    file << "\" name=\"" << plugin_name << "\"></plugin></plugins></ie>";
    file.flush();
    file.close();
}

static void remove_plugin_xml(const std::string& file_name) {
    ov::test::utils::removeFile(file_name);
}

TEST(CoreBaseTest, LoadPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getOpenvinoLibDirectory() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_name));
    remove_plugin_xml(xml_file_path);
}

TEST(CoreBaseTest, LoadPluginDifferentXMLExtension) {
    std::string xml_file_name = "test_plugin.test";
    std::string xml_file_path =
        ov::test::utils::getOpenvinoLibDirectory() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_name));
    remove_plugin_xml(xml_file_path);
}

TEST(CoreBaseTest, LoadAbsoluteOVPathPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getOpenvinoLibDirectory() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_path));
    remove_plugin_xml(xml_file_path);
}

TEST(CoreBaseTest, LoadAbsoluteCWPathPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getCurrentWorkingDir() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_path));
    remove_plugin_xml(xml_file_path);
}

TEST(CoreBaseTest, LoadRelativeCWPathPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getCurrentWorkingDir() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_name));
    remove_plugin_xml(xml_file_path);
}

TEST(CoreBaseTest, LoadOVFolderOverCWPathPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string cwd_file_path =
        ov::test::utils::getCurrentWorkingDir() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    std::string ov_file_path =
        ov::test::utils::getOpenvinoLibDirectory() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(cwd_file_path);
    create_plugin_xml(ov_file_path, "2");
    ov::Core core(xml_file_name);
    auto version = core.get_versions("2");
    EXPECT_EQ(1, version.size());
    version = core.get_versions("1");
    EXPECT_EQ(0, version.size());
    remove_plugin_xml(cwd_file_path);
    remove_plugin_xml(ov_file_path);
}

#    if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
TEST(CoreBaseTest, ReadModelwithSymlink) {
    fs::create_directory("test_link");
    std::string modelName = "test_link/test.xml";
    std::string weightsName = "test_link/test.bin";
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(modelName, weightsName);
    manager.run_passes(ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::Type_t::f32));

    std::string modelNameSymlink = "test_link/test_symlink.xml";
    fs::create_symlink(modelName, modelNameSymlink);
    ov::Core core;
    EXPECT_NO_THROW(core.read_model(modelName, weightsName));
    EXPECT_THROW(core.read_model(modelNameSymlink, weightsName), std::runtime_error);

    fs::remove_all("test_link");
    ASSERT_FALSE(ov::util::directory_exists("test_link"));
}
#    endif
#endif
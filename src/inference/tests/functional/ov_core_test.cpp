// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

#ifndef OPENVINO_STATIC_LIBRARY

static void create_plugin_xml(const std::string& file_name) {
    std::ofstream file(file_name);

    file << "<ie><plugins><plugin location=\"";
    file << ov::test::utils::getExecutableDirectory();
    file << ov::util::FileTraits<char>::file_separator;
    file << ov::util::FileTraits<char>::library_prefix();
    file << "mock_engine";
    file << OV_BUILD_POSTFIX;
    file << ov::util::FileTraits<char>::dot_symbol;
    file << ov::util::FileTraits<char>::library_ext();
    file << "\" name=\"1\"></plugin></plugins></ie>";
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

TEST(CoreBaseTest, LoadPluginInvalidXML) {
    std::string xml_file_name = "test_plugin.test";
    std::string xml_file_path =
        ov::test::utils::getOpenvinoLibDirectory() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_ANY_THROW(ov::Core core(xml_file_name));
    remove_plugin_xml(xml_file_path);
}

TEST(CoreBaseTest, LoadAbsolutePathPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getCurrentWorkingDir() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_path));
    remove_plugin_xml(xml_file_path);
}

#endif
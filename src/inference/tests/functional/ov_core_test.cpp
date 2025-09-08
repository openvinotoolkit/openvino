// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

class CoreBaseTest : public testing::Test {
protected:
    void generate_test_model_files(const std::string& name) {
        auto prefix = ov::test::utils::generateTestFilePrefix();
        model_file_name = prefix + name + ".xml";
        weight_file_name = prefix + name + ".bin";
        ov::test::utils::generate_test_model(model_file_name, weight_file_name);
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
        for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
            std::wstring postfix_w = ov::util::string_to_wstring(prefix) + L"_" +
                                     ov::test::utils::test_unicode_postfix_vector[testIndex] +
                                     ov::util::string_to_wstring(name);
            auto model_file_path_w = postfix_w + L".xml";
            auto weight_file_path_w = postfix_w + L".bin";
            ov::test::utils::generate_test_model(model_file_path_w, weight_file_path_w);
            model_files_name_w.push_back(model_file_path_w);
            weight_files_name_w.push_back(weight_file_path_w);
        }
#endif
    }

    void TearDown() override {
        ov::test::utils::removeIRFiles(model_file_name, weight_file_name);
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
        for (auto& m_path_w : model_files_name_w) {
            ov::test::utils::removeFile(m_path_w);
        }
        for (auto& w_path_w : weight_files_name_w) {
            ov::test::utils::removeFile(w_path_w);
        }
#endif
    }

    std::string model_file_name, weight_file_name;
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    std::vector<std::wstring> model_files_name_w, weight_files_name_w;
#endif
};

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

TEST_F(CoreBaseTest, LoadPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getOpenvinoLibDirectory() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_name));
    remove_plugin_xml(xml_file_path);
}

TEST_F(CoreBaseTest, LoadPluginDifferentXMLExtension) {
    std::string xml_file_name = "test_plugin.test";
    std::string xml_file_path =
        ov::test::utils::getOpenvinoLibDirectory() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_name));
    remove_plugin_xml(xml_file_path);
}

TEST_F(CoreBaseTest, LoadAbsoluteOVPathPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getOpenvinoLibDirectory() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_path));
    remove_plugin_xml(xml_file_path);
}

TEST_F(CoreBaseTest, LoadAbsoluteCWPathPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getCurrentWorkingDir() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_path));
    remove_plugin_xml(xml_file_path);
}

TEST_F(CoreBaseTest, LoadRelativeCWPathPluginXML) {
    std::string xml_file_name = "test_plugin.xml";
    std::string xml_file_path =
        ov::test::utils::getCurrentWorkingDir() + ov::util::FileTraits<char>::file_separator + xml_file_name;
    create_plugin_xml(xml_file_path);
    EXPECT_NO_THROW(ov::Core core(xml_file_name));
    remove_plugin_xml(xml_file_path);
}

TEST_F(CoreBaseTest, LoadOVFolderOverCWPathPluginXML) {
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

#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
TEST_F(CoreBaseTest, read_model_with_std_fs_path_unicode) {
    generate_test_model_files("test-model");
    ov::Core core;
    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        std::filesystem::path model_path = model_files_name_w[testIndex];
        std::filesystem::path weight_path = weight_files_name_w[testIndex];
        {
            const auto model = core.read_model(model_path);
            EXPECT_NE(model, nullptr);
        }
        {
            const auto model = core.read_model(model_path, weight_path);
            EXPECT_NE(model, nullptr);
        }
    }
}
#endif
namespace ov::test {
TEST_F(CoreBaseTest, read_model_with_std_fs_path) {
    generate_test_model_files("test-model");

    const auto model_path = std::filesystem::path(model_file_name);
    const auto weight_path = std::filesystem::path(weight_file_name);

    ov::Core core;
    {
        const auto model = core.read_model(model_path);
        EXPECT_NE(model, nullptr);
    }
    {
        const auto model = core.read_model(model_path, weight_path);
        EXPECT_NE(model, nullptr);
    }
}

TEST_F(CoreBaseTest, compile_model_with_std_fs_path) {
    generate_test_model_files("model2");

    const auto model_path = std::filesystem::path(model_file_name);
    const auto weight_path = std::filesystem::path(weight_file_name);

    ov::Core core;
    {
        const auto model = core.compile_model(model_path);
        EXPECT_TRUE(model);
    }
    {
        const auto devices = core.get_available_devices();

        const auto model = core.compile_model(model_path, devices.at(0), ov::AnyMap{});
        EXPECT_TRUE(model);
    }
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        std::filesystem::path model_path_w = model_files_name_w[testIndex];
        {
            const auto model = core.compile_model(model_path_w);
            EXPECT_TRUE(model);
        }
        {
            const auto devices = core.get_available_devices();

            const auto model = core.compile_model(model_path_w, devices.at(0), ov::AnyMap{});
            EXPECT_TRUE(model);
        }
    }
#endif
}
}  // namespace ov::test

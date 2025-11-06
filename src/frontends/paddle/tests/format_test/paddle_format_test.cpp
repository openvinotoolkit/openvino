// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

#include "openvino/frontend/paddle/decoder.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/paddle/utils.hpp"
#include "decoder_json.hpp"
#include "yaml_metadata.hpp"

using namespace testing;
using namespace ov;
using namespace ov::frontend;
using namespace ov::frontend::paddle;

class PaddleFormatTest : public ::testing::Test {
protected:
    void SetUp() override {
    FrontEndManager fem;
    m_frontend = fem.load_by_framework("paddle");
        m_test_dir = std::filesystem::temp_directory_path() / "paddle_format_test";
        std::filesystem::create_directories(m_test_dir);
    }

    void TearDown() override {
        std::filesystem::remove_all(m_test_dir);
    }

    void create_dummy_files(const std::string& format) {
        if (format == "new") {
            // Create PP-OCRv5 format files
            std::ofstream json_file(m_test_dir / "inference.json");
            json_file << R"({
                "ops": [
                    {
                        "type": "conv2d",
                        "inputs": {"x": "input"},
                        "outputs": {"y": "output"},
                        "attrs": {"kernel_size": [3, 3]}
                    }
                ],
                "version": "2.3"
            })";
            json_file.close();

            std::ofstream yml_file(m_test_dir / "inference.yml");
            yml_file << R"(
model_type: ocr
model_specs:
  architecture: PP-OCRv5
  det_arch: DBNet
)";
            yml_file.close();

            std::ofstream params_file(m_test_dir / "inference.pdiparams");
            params_file << "dummy params";
            params_file.close();
        } else {
            // Create legacy format files
            std::ofstream model_file(m_test_dir / "inference.pdmodel");
            model_file << "dummy model";
            model_file.close();

            std::ofstream params_file(m_test_dir / "inference.pdiparams");
            params_file << "dummy params";
            params_file.close();

            std::ofstream info_file(m_test_dir / "inference.pdiparams.info");
            info_file << "dummy info";
            info_file.close();
        }
    }

    std::shared_ptr<ov::frontend::FrontEnd> m_frontend;
    std::filesystem::path m_test_dir;
};

TEST_F(PaddleFormatTest, DetectPPOCRv5Format) {
    create_dummy_files("new");
    EXPECT_TRUE(is_new_paddle_format(m_test_dir.string()));
    
    auto [json_path, yml_path, params_path] = get_model_files(m_test_dir.string());
    EXPECT_EQ(json_path, (m_test_dir / "inference.json").string());
    EXPECT_EQ(yml_path, (m_test_dir / "inference.yml").string());
    EXPECT_EQ(params_path, (m_test_dir / "inference.pdiparams").string());
}

TEST_F(PaddleFormatTest, DetectLegacyFormat) {
    create_dummy_files("legacy");
    EXPECT_FALSE(is_new_paddle_format(m_test_dir.string()));
    
    auto [model_path, yml_path, params_path] = get_model_files(m_test_dir.string());
    EXPECT_EQ(model_path, (m_test_dir / "inference.pdmodel").string());
    EXPECT_TRUE(yml_path.empty());
    EXPECT_EQ(params_path, (m_test_dir / "inference.pdiparams").string());
}

TEST_F(PaddleFormatTest, LoadJSONModel) {
    create_dummy_files("new");
    std::string json_path = (m_test_dir / "inference.json").string();

    // Test JSON decoder
    auto decoder = std::make_shared<DecoderJSON>(json_path);
    EXPECT_NE(decoder, nullptr);

    // Basic validation of parsed model
    EXPECT_EQ(decoder->get_op_size(), 1);  // One conv2d operator
    const auto& op = decoder->get_op(0);
    EXPECT_EQ(op.type, "conv2d");
}

TEST_F(PaddleFormatTest, LoadYAMLMetadata) {
    create_dummy_files("new");
    std::string yml_path = (m_test_dir / "inference.yml").string();

    // Test YAML metadata reader
    YAMLMetadataReader yaml_reader(yml_path);
    auto metadata = yaml_reader.get_metadata();

    EXPECT_EQ(metadata["model_type"], "ocr");
    EXPECT_EQ(metadata["architecture"], "PP-OCRv5");
    EXPECT_EQ(metadata["det_arch"], "DBNet");
}
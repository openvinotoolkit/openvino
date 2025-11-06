// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>

#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/paddle/decoder.hpp"
#include "openvino/frontend/paddle/utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::frontend;
using namespace ov::frontend::paddle;

class PaddleFormatTest : public ::testing::Test {
protected:
    void SetUp() override {
    FrontEndManager fem;
    m_frontend = fem.load_by_framework("paddle");
    }

    std::shared_ptr<ov::frontend::FrontEnd> m_frontend;
};

TEST_F(PaddleFormatTest, DetectPPOCRv5Format) {
    const std::string test_dir = "test_models/pp_ocrv5";
    
    // Create test directory with PP-OCRv5 files
    std::filesystem::create_directories(test_dir);
    
    std::ofstream json_file(test_dir + "/inference.json");
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

    std::ofstream yml_file(test_dir + "/inference.yml");
    yml_file << R"(
model_type: ocr
model_specs:
  architecture: PP-OCRv5
  det_arch: DBNet
)";
    yml_file.close();

    // Verify format detection
    EXPECT_TRUE(is_new_paddle_format(test_dir));
    
    // Test model loading
    auto [model_file, yml_path, params_file] = get_model_files(test_dir);
    EXPECT_EQ(model_file, test_dir + "/inference.json");
    EXPECT_EQ(yml_path, test_dir + "/inference.yml");
    EXPECT_EQ(params_file, test_dir + "/inference.pdiparams");

    // Clean up
    std::filesystem::remove_all(test_dir);
}

TEST_F(PaddleFormatTest, DetectLegacyFormat) {
    const std::string test_dir = "test_models/legacy";
    
    // Create test directory with legacy files
    std::filesystem::create_directories(test_dir);
    
    std::ofstream model_file(test_dir + "/inference.pdmodel");
    model_file << "dummy pdmodel content";
    model_file.close();

    std::ofstream params_file(test_dir + "/inference.pdiparams");
    params_file << "dummy params content";
    params_file.close();

    // Verify format detection
    EXPECT_FALSE(is_new_paddle_format(test_dir));
    
    // Test model loading
    auto [model_path, yml_path, params_path] = get_model_files(test_dir);
    EXPECT_EQ(model_path, test_dir + "/inference.pdmodel");
    EXPECT_TRUE(yml_path.empty()); // No yml file in legacy format
    EXPECT_EQ(params_path, test_dir + "/inference.pdiparams");

    // Clean up
    std::filesystem::remove_all(test_dir);
}

// Add more tests for model loading and conversion
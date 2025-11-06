// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <string>

namespace ov {
namespace frontend {
namespace paddle {

namespace fs = std::filesystem;

// Detect if we're dealing with PP-OCRv5 format
inline bool is_new_paddle_format(const std::string& model_path) {
    fs::path path(model_path);
    if (fs::is_directory(path)) {
        return fs::exists(path / "inference.json");
    }
    return false;
}

// Get file paths based on format type
inline std::tuple<std::string, std::string, std::string> get_model_files(const std::string& model_path) {
    fs::path path(model_path);
    
    if (is_new_paddle_format(model_path)) {
        // PP-OCRv5 format (new)
        return {
            (path / "inference.json").string(),
            (path / "inference.yml").string(),
            (path / "inference.pdiparams").string()
        };
    } else {
        // Legacy format
        // Get model and parameter paths
        std::string model_file = (path / "inference.pdmodel").string();
        std::string params_file = (path / "inference.pdiparams").string();
        
        return {model_file, "", params_file}; // Empty string for yml since it doesn't exist in legacy format
    }
}

} // namespace paddle
} // namespace frontend
} // namespace ov
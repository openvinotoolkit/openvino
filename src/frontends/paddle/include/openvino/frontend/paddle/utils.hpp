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
        std::string model_file = path.string();
        // portable ends_with replacement for C++ standard library compatibility
        auto ends_with = [](const std::string& s, const std::string& suffix) {
            if (s.size() < suffix.size())
                return false;
            return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

        // First check for inference.pdmodel, then __model__ for legacy format
        fs::path pdmodel_path = path / "inference.pdmodel";
        model_file = pdmodel_path.string();

        // Handle parameters file
        std::string params_file = (path / "inference.pdiparams").string();
        std::string info_file = params_file + ".info";
        
        return {model_file, "", params_file}; // Empty string for yml since it doesn't exist in legacy format
    }
}

} // namespace paddle
} // namespace frontend
} // namespace ov
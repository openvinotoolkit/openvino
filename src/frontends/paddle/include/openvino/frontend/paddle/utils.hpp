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

/**
 * @brief Detect if the model is in the new PaddlePaddle 3.0 (PP-OCRv5) JSON format
 *
 * Checks if the given path is a directory containing an inference.json file,
 * which indicates the new JSON-based model format introduced in PaddlePaddle 3.0.
 *
 * @param model_path Path to the model directory or file
 * @return true if the path is a directory containing inference.json, false otherwise
 */
inline bool is_new_paddle_format(const std::string& model_path) {
    try {
        fs::path path(model_path);

        // Check if path exists
        if (!fs::exists(path)) {
            return false;
        }

        // Check if it's a directory and contains inference.json
        if (fs::is_directory(path)) {
            fs::path json_file = path / "inference.json";
            return fs::exists(json_file) && fs::is_regular_file(json_file);
        }

        return false;
    } catch (const fs::filesystem_error&) {
        // Return false on any filesystem error (permissions, invalid path, etc.)
        return false;
    }
}

/**
 * @brief Get file paths for model components based on format type
 *
 * Returns a tuple of (model_file, yml_file, params_file) paths:
 * - For PP-OCRv5 format: (inference.json, inference.yml, inference.pdiparams)
 * - For legacy format: (inference.pdmodel, "", inference.pdiparams)
 *
 * @param model_path Path to the model directory
 * @return Tuple of (model_file_path, yml_file_path, params_file_path)
 * @note The returned paths are not validated for existence
 */
inline std::tuple<std::string, std::string, std::string> get_model_files(const std::string& model_path) {
    fs::path path(model_path);

    if (is_new_paddle_format(model_path)) {
        // PP-OCRv5 format (new)
        return {(path / "inference.json").string(),
                (path / "inference.yml").string(),
                (path / "inference.pdiparams").string()};
    } else {
        // Legacy format
        return {(path / "inference.pdmodel").string(),
                "",  // No YAML file in legacy format
                (path / "inference.pdiparams").string()};
    }
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
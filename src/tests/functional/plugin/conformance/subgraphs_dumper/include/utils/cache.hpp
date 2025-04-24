// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <regex>
#include <fstream>

#include "openvino/openvino.hpp"
#include "cache/cache.hpp"

namespace ov {
namespace util {

const std::shared_ptr<ov::Core> core = std::make_shared<ov::Core>();

static std::vector<std::regex> FROTEND_REGEXP = {
#ifdef ENABLE_OV_ONNX_FRONTEND
    std::regex(R"(.*\.onnx)"),
#endif
#ifdef ENABLE_OV_PADDLE_FRONTEND
    std::regex(R"(.*\.pdmodel)"),
    std::regex(R"(.*__model__)"),
#endif
#ifdef ENABLE_OV_TF_FRONTEND
    std::regex(R"(.*\.pb)"),
#endif
#ifdef ENABLE_OV_IR_FRONTEND
    std::regex(R"(.*\.xml)"),
#endif
#ifdef ENABLE_OV_TF_LITE_FRONTEND
    std::regex(R"(.*\.tflite)"),
#endif
#ifdef ENABLE_OV_PYTORCH_FRONTEND
    std::regex(R"(.*\.pt)"),
#endif
};

enum ModelCacheStatus {
    SUCCEED = 0,
    NOT_FULLY_CACHED = 1,
    NOT_READ = 2,
    LARGE_MODELS_EXCLUDED = 3,
    LARGE_MODELS_INCLUDED = 4,
};

static std::map<ModelCacheStatus, std::string> model_cache_status_to_str = {
    { ModelCacheStatus::SUCCEED, "successful_models" },
    { ModelCacheStatus::NOT_FULLY_CACHED, "not_fully_cached_models" },
    { ModelCacheStatus::NOT_READ, "not_read_models" },
    { ModelCacheStatus::LARGE_MODELS_EXCLUDED, "large_models_excluded" },
    { ModelCacheStatus::LARGE_MODELS_INCLUDED, "large_models_included" },
};

std::pair<std::vector<std::string>, std::pair<ModelCacheStatus, std::vector<std::string>>>
find_models(const std::vector<std::string> &dirs, const std::string& regexp = ".*");

// model_cache_status: model_list
std::map<ModelCacheStatus, std::vector<std::string>> cache_models(
    std::shared_ptr<ov::tools::subgraph_dumper::ICache>& cache,
    const std::vector<std::string>& models,
    bool extract_body, bool from_cache = false);

void save_model_status_to_file(const std::map<ModelCacheStatus, std::vector<std::string>>& caching_status,
                               const std::string& output_dir);



template <typename ElementType>
inline void vector_to_file(const std::vector<ElementType>& vec, const std::string& output_file_path) {
    std::ofstream output_file;
    output_file.open(output_file_path, std::ios::out | std::ios::trunc);
    for (const auto& element : vec) {
        output_file << element << std::endl;
    }
    output_file.close();
}

}  // namespace util
}  // namespace ov

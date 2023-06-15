// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <regex>

#include "openvino/util/file_util.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

#include "cache/cache.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

static std::vector<std::regex> FROTEND_REGEXP {
#ifdef ENABLE_OV_ONNX_FRONTEND
    std::regex(R"(.*\.onnx)",
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
    NOT_READ = 2
};

static std::map<ModelCacheStatus, std::string> model_cache_status_to_str = {
    { ModelCacheStatus::SUCCEED, "successful_models" },
    { ModelCacheStatus::NOT_FULLY_CACHED, "not_fully_cached_models" },
    { ModelCacheStatus::NOT_READ, "not_read_models" },
};

inline std::vector<std::string> find_models(const std::vector<std::string> &dirs, const std::string& regexp = ".*") {
    std::vector<std::string> models, full_content;
    for (const auto& dir : dirs) {
        std::vector<std::string> dir_content;
        if (ov::util::directory_exists(dir)) {
            dir_content = CommonTestUtils::getFileListByPatternRecursive({dir}, FROTEND_REGEXP);
        } else if (ov::util::file_exists(dir) && std::regex_match(dir, std::regex(".*" + std::string(CommonTestUtils::LST_EXTENSION)))) {
            dir_content = CommonTestUtils::readListFiles({dir});
        } else {
            std::string msg = "Input directory (" + dir + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        if (!dir_content.empty()) {
            full_content.insert(full_content.end(), dir_content.begin(), dir_content.end());
        }
    }
    auto in_regex = std::regex(regexp);
    for (const auto& file : full_content) {
        if (std::regex_match(file, in_regex)) {
            try {
                models.emplace_back(file);
            } catch (std::exception& e) {
                std::cout << "Impossible to read model: " << file << std::endl << "Exception: " << e.what();
            }
        }
    }
    return models;
}

// model_cache_status: model_list
inline std::map<ModelCacheStatus, std::vector<std::string>> cache_models(
    std::vector<std::shared_ptr<ICache>>& caches,
    const std::vector<std::string>& models,
    bool extract_body) {
    std::map<ModelCacheStatus, std::vector<std::string>> cache_status = {
        { ModelCacheStatus::SUCCEED, {} },
        { ModelCacheStatus::NOT_FULLY_CACHED, {} },
        { ModelCacheStatus::NOT_READ, {} }
    };
    auto core = ov::test::utils::PluginCache::get().core();

    for (const auto& model : models) {
        if (ov::util::file_exists(model)) {
            std::cout << "Processing model: " << model << std::endl;
            ModelCacheStatus model_status = ModelCacheStatus::SUCCEED;
            try {
                std::shared_ptr<ov::Model> function = core->read_model(model);
                try {
                    for (auto& cache : caches) {
                        cache->update_cache(function, model, extract_body);
                    }
                } catch (std::exception &e) {
                    std::cout << "Model processing failed with exception:" << std::endl << e.what() << std::endl;
                    model_status = ModelCacheStatus::NOT_FULLY_CACHED;
                }
            } catch (std::exception &e) {
                model_status = ModelCacheStatus::NOT_READ;
                std::cout << "Model reading failed with exception:" << std::endl << e.what() << std::endl;
            }
            cache_status[model_status].push_back(model);
        }
    }
    return cache_status;
}

inline void save_model_status_to_file(const std::map<ModelCacheStatus, std::vector<std::string>>& caching_status, const std::string& output_dir) {
    std::string cache_status_path = ov::util::path_join({output_dir, "model_caching_status"});
    if (!ov::util::directory_exists(cache_status_path)) {
        ov::util::create_directory_recursive(cache_status_path);
    }
    for (const auto& status_info : caching_status) {
        std::string output_file_path = ov::util::path_join({ cache_status_path, model_cache_status_to_str[status_info.first] + CommonTestUtils::LST_EXTENSION});
        CommonTestUtils::vec2File(status_info.second, output_file_path);
    }
}

inline void serialize_cache(const std::vector<std::shared_ptr<ICache>>& caches, const std::string& serilization_dir) {
    for (auto& cache : caches) {
        cache->set_serialization_dir(serilization_dir);
        cache->serialize_cache();
    }
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <regex>
#include <unordered_set>

#include "openvino/util/file_util.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

#include "cache/cache.hpp"
#include "utils/node.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

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
    NOT_READ = 2
};

static std::map<ModelCacheStatus, std::string> model_cache_status_to_str = {
    { ModelCacheStatus::SUCCEED, "successful_models" },
    { ModelCacheStatus::NOT_FULLY_CACHED, "not_fully_cached_models" },
    { ModelCacheStatus::NOT_READ, "not_read_models" },
};

std::vector<std::string> find_models(const std::vector<std::string> &dirs, const std::string& regexp = ".*");

// model_cache_status: model_list
std::map<ModelCacheStatus, std::vector<std::string>> cache_models(
    std::vector<std::shared_ptr<ICache>>& caches,
    const std::vector<std::string>& models,
    bool extract_body);

void save_model_status_to_file(const std::map<ModelCacheStatus, std::vector<std::string>>& caching_status,
                               const std::string& output_dir);

std::pair<std::shared_ptr<ov::Model>, std::map<std::string, InputInfo>>
generate_model(const std::set<std::shared_ptr<ov::Node>>& nodes,
               const std::shared_ptr<ov::Node>& start_node,
               std::unordered_set<std::string>& checked_ops);

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov

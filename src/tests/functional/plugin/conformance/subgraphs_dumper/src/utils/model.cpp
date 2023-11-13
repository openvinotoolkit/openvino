// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_util.hpp"
#include "utils/model.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

void save_model_status_to_file(const std::map<ModelCacheStatus, std::vector<std::string>>& caching_status,
                               const std::string& output_dir) {
    std::string cache_status_path = ov::util::path_join({output_dir, "model_caching_status"});
    if (!ov::util::directory_exists(cache_status_path)) {
        ov::util::create_directory_recursive(cache_status_path);
    }
    for (const auto& status_info : caching_status) {
        std::string output_file_path = ov::util::path_join({ cache_status_path, model_cache_status_to_str[status_info.first] + ov::test::utils::LST_EXTENSION});
        ov::test::utils::vec2File(status_info.second, output_file_path);
    }
}

// { models, { not_read_model }}
std::pair<std::vector<std::string>, std::pair<ModelCacheStatus, std::vector<std::string>>>
find_models(const std::vector<std::string> &dirs, const std::string& regexp) {
    std::vector<std::string> models, full_content, not_read_model;
    for (const auto& dir : dirs) {
        std::vector<std::string> dir_content;
        if (ov::util::directory_exists(dir)) {
            dir_content = ov::test::utils::getFileListByPatternRecursive({dir}, FROTEND_REGEXP);
        } else if (ov::util::file_exists(dir) && std::regex_match(dir, std::regex(".*" + std::string(ov::test::utils::LST_EXTENSION)))) {
            dir_content = ov::test::utils::readListFiles({dir});
        } else {
            std::cout << "[ ERROR ] Input directory (" << dir << ") doesn't not exist!" << std::endl;
        }
        if (!dir_content.empty()) {
            full_content.insert(full_content.end(), dir_content.begin(), dir_content.end());
        }
    }
    std::multimap<size_t, std::string> models_sorted_by_size;
    auto in_regex = std::regex(regexp);
    for (const auto& model_file : full_content) {
        if (std::regex_match(model_file, in_regex)) {
            try {
                // models.emplace_back(file);
                if (ov::util::file_exists(model_file)) {
                    auto model_size = core->read_model(model_file)->get_graph_size();
                    models_sorted_by_size.insert({ model_size, model_file});
                } else {
                    continue;
                }
            } catch (std::exception) {
                not_read_model.emplace_back(model_file);
                // std::cout << "[ ERROR ] Impossible to read model: " << model_file << std::endl << "Exception: " << e.what();
            }
        }
    }
    // sort model by size with reverse
    auto model_cnt = models_sorted_by_size.size();
    models.resize(model_cnt);
    auto it = models_sorted_by_size.rbegin();
    for (size_t i = 0; i < model_cnt; ++i) {
        models[i] = it->second;
        ++it;
    }
    std::cout << "[ INFO ] Total model number is " << models.size() << std::endl;
    return { models, { ModelCacheStatus::NOT_READ, not_read_model } };
}

std::string get_model_type(const std::shared_ptr<ov::Model>& model) {
    if (is_dynamic_model(model)) {
        return "dynamic";
    }
    return "static";
}

std::map<ModelCacheStatus, std::vector<std::string>> cache_models(
    std::shared_ptr<ICache>& cache,
    const std::vector<std::string>& models,
    bool extract_body, bool from_cache) {
    std::map<ModelCacheStatus, std::vector<std::string>> cache_status = {
        { ModelCacheStatus::SUCCEED, {} },
        { ModelCacheStatus::NOT_FULLY_CACHED, {} },
        { ModelCacheStatus::NOT_READ, {} },
        { ModelCacheStatus::LARGE_MODELS_EXCLUDED, {} },
        { ModelCacheStatus::LARGE_MODELS_INCLUDED, {} },
    };
    auto models_size = models.size();

    for (size_t i = 0; i < models_size; ++i) {
        const auto& model = models[i];

        if (ov::util::file_exists(model)) {
            std::cout << "[ INFO ][ " << i + 1 << "/" << models_size << " ] model will be processed" << std::endl;
            ModelCacheStatus model_status = ModelCacheStatus::SUCCEED;
            try {
                std::shared_ptr<ov::Model> function = core->read_model(model);
                try {
                    if (cache->is_model_large_to_read(function, model)) {
                        cache_status[ModelCacheStatus::LARGE_MODELS_EXCLUDED].push_back(model);
                        continue;
                    } else if (cache->is_model_large_to_store_const(function)) {
                        cache_status[ModelCacheStatus::LARGE_MODELS_INCLUDED].push_back(model);
                    }
                    cache->update_cache(function, model, extract_body, from_cache);
                } catch (std::exception& e) {
                    std::cout << "[ ERROR ] Model processing failed with exception:" << std::endl << e.what() << std::endl;
                    model_status = ModelCacheStatus::NOT_FULLY_CACHED;
                }
            } catch (std::exception) {
                model_status = ModelCacheStatus::NOT_READ;
                // std::cout << "[ ERROR ] Model reading failed with exception:" << std::endl << e.what() << std::endl;
            }
            cache_status[model_status].push_back(model);
        }
    }

    return cache_status;
}

std::map<std::string, InputInfo>
get_input_info_by_model(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, InputInfo> in_info;
    for (const auto& node : model->get_ordered_ops()) {
        InputInfo::Range ranges(DEFAULT_MIN_VALUE, DEFAULT_MAX_VALUE);
        bool is_const = false;
        if (ov::op::util::is_constant(node)) {
            std::shared_ptr<ov::op::v0::Constant> constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
            auto const_ranges = get_const_ranges(constant,
                                                 constant->get_default_output().get_element_type());
            ranges = const_ranges;
        } else if (!ov::op::util::is_parameter(node)) {
            continue;
        }
        auto partial_shape = node->get_default_output().get_partial_shape();
        in_info.insert({node->get_friendly_name(),
                        InputInfo(partial_shape, ranges.min, ranges.max, is_const)});
    }
    return in_info;
}

std::map<std::string, InputInfo>
align_input_info(const std::shared_ptr<ov::Model>& model,
                 const std::shared_ptr<ov::Model>& model_ref,
                 const std::map<std::string, InputInfo>& in_info,
                 const std::map<std::string, InputInfo>& in_info_ref,
                 const std::map<std::string, std::string> &matched_op) {
    bool is_update_required = !matched_op.empty();
    if (!is_update_required) {
        for (const auto& ref_item : in_info_ref) {
            if (!in_info.count(ref_item.first)) {
                is_update_required = true;
                break;
            } else if (in_info.at(ref_item.first).is_const != ref_item.second.is_const) {
                throw std::runtime_error("Impossible to update input info!!!");
            }
        }
    }

    std::map<std::string, InputInfo> updated_input_info = in_info_ref;
    if (is_update_required) {
        // align matched model names
        const auto& ref_model_ops = model_ref->get_ordered_ops();
        const auto& model_ops = model->get_ordered_ops();
        size_t ref_ordered_ops_size = ref_model_ops.size();
        size_t ordered_ops_size = model_ops.size();
        if (ref_ordered_ops_size != ordered_ops_size && matched_op.empty()) {
            throw std::runtime_error("Matched models can not be compared according different op numbers!");
        }
        for (size_t i = 0; i < ordered_ops_size; ++i) {
            auto model_op_name = model_ops[i]->get_friendly_name();
            if (!in_info.count(model_op_name)) {
                continue;
            }
            if (!matched_op.empty()) {
                if (!matched_op.count(model_op_name)) {
                    continue;
                }
            }
            auto model_ref_op_name = matched_op.empty() ? ref_model_ops[i]->get_friendly_name() : matched_op.at(model_op_name);

            const auto& in_info_item = in_info.at(model_op_name);
            const auto& ref_in_info_item = in_info_ref.at(model_ref_op_name);
            if (in_info_item.is_const != ref_in_info_item.is_const) {
                throw std::runtime_error("Impossible to update input info!!!");
            }
            updated_input_info[model_ref_op_name] = in_info_item;
        }
    }
    return updated_input_info;
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
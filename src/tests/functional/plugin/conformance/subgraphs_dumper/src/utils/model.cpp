// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/model.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

inline std::unordered_map<std::string, std::shared_ptr<ov::Node>>
update_nodes(const std::set<std::shared_ptr<ov::Node>>& nodes,
             const std::shared_ptr<ov::Node>& start_node) {
    std::unordered_map<std::string, std::shared_ptr<ov::Node>> model_map;
    std::shared_ptr<ov::Node> cloned_op = nullptr;

    for (const auto& op : nodes) {
        if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) ||
            ov::op::util::is_output(op)) {
            continue;
        }
        cloned_op = clone_node(op, true, false, "Op_" + std::to_string(model_map.size()));
        model_map.insert({ op->get_friendly_name(), cloned_op });
    }

    for (const auto& op : nodes) {
        if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op) ||
            ov::op::util::is_output(op)) {
            continue;
        }
        auto op_name = op->get_friendly_name();
        cloned_op = model_map[op->get_friendly_name()];
        size_t inputs_size = op->inputs().size();
        ov::OutputVector in_out_vector(inputs_size);
        int filled_input_idx = -1;
        for (size_t in_idx = 0; in_idx < inputs_size; ++in_idx) {
            auto in_node = op->get_input_node_ptr(in_idx)->shared_from_this();
            for (size_t in_out_idx = 0; in_out_idx < in_node->outputs().size(); ++in_out_idx) {
                for (const auto& target_input : in_node->output(in_out_idx).get_target_inputs()) {
                    auto out_in_node = target_input.get_node()->shared_from_this();
                    if (out_in_node == op) {
                        auto in_node_name = in_node->get_friendly_name();
                        in_out_vector[in_idx] = model_map.count(in_node_name) ?
                                                model_map.at(in_node_name)->output(in_out_idx) :
                                                cloned_op->get_input_node_ptr(in_idx)->output(0);
                        if (model_map.count(in_node_name)) {
                            filled_input_idx++;
                        }
                        break;
                    }
                }
                if (filled_input_idx == in_idx) {
                    break;
                }
            }
        }
        // todo: iefode: check this code
        if (filled_input_idx < 0 && op_name != start_node->get_friendly_name()) {
            model_map.erase(op_name);
        } else if (filled_input_idx >= 0) {
            auto name = cloned_op->get_friendly_name();
            model_map[op_name] = cloned_op->clone_with_new_inputs(in_out_vector);
            model_map[op_name]->set_friendly_name(name);
        }
    }
    return model_map;
}

std::pair<std::shared_ptr<ov::Model>, std::map<std::string, InputInfo>>
generate_model(const std::set<std::shared_ptr<ov::Node>>& nodes,
               const std::shared_ptr<ov::Node>& start_node,
               std::unordered_set<std::string>& checked_ops) {
    if (nodes.size() < 2) {
        throw std::runtime_error("Incorrect node number to create model");
    }
    auto model_map = update_nodes(nodes, start_node);
    if (model_map.size() < 2) {
        throw std::runtime_error("Incorrect node number to create model");
    }
    ov::OutputVector results;
    std::map<std::string, InputInfo> input_info;
    for (const auto& op : model_map) {
        checked_ops.insert(op.first);
        auto this_input_info = get_input_info_by_node(op.second);
        input_info.insert(this_input_info.begin(), this_input_info.end());
        for (size_t j = 0; j < op.second->outputs().size(); ++j) {
            if (op.second->output(j).get_target_inputs().empty()) {
                results.push_back(std::make_shared<ov::op::v0::Result>(op.second->output(j)));
            }
        }
    }
    return { std::make_shared<ov::Model>(results), input_info };
}

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

std::vector<std::string> find_models(const std::vector<std::string> &dirs, const std::string& regexp) {
    std::vector<std::string> models, full_content;
    for (const auto& dir : dirs) {
        std::vector<std::string> dir_content;
        if (ov::util::directory_exists(dir)) {
            dir_content = ov::test::utils::getFileListByPatternRecursive({dir}, FROTEND_REGEXP);
        } else if (ov::util::file_exists(dir) && std::regex_match(dir, std::regex(".*" + std::string(ov::test::utils::LST_EXTENSION)))) {
            dir_content = ov::test::utils::readListFiles({dir});
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
                std::cout << "[ ERROR ] Impossible to read model: " << file << std::endl << "Exception: " << e.what();
            }
        }
    }
    return models;
}

std::map<ModelCacheStatus, std::vector<std::string>> cache_models(
    std::vector<std::shared_ptr<ICache>>& caches,
    const std::vector<std::string>& models,
    bool extract_body) {
    std::map<ModelCacheStatus, std::vector<std::string>> cache_status = {
        { ModelCacheStatus::SUCCEED, {} },
        { ModelCacheStatus::NOT_FULLY_CACHED, {} },
        { ModelCacheStatus::NOT_READ, {} }
    };
    auto core = ov::test::utils::PluginCache::get().core();

    for (auto& cache : caches) {
        for (const auto& model : models) {
            if (ov::util::file_exists(model)) {
                ModelCacheStatus model_status = ModelCacheStatus::SUCCEED;
                try {
                    std::shared_ptr<ov::Model> function = core->read_model(model);
                    try {
                        for (auto& cache : caches) {
                            cache->update_cache(function, model, extract_body);
                        }
                    } catch (std::exception &e) {
                        std::cout << "[ ERROR ] Model processing failed with exception:" << std::endl << e.what() << std::endl;
                        model_status = ModelCacheStatus::NOT_FULLY_CACHED;
                    }
                } catch (std::exception &e) {
                    model_status = ModelCacheStatus::NOT_READ;
                    std::cout << "[ ERROR ] Model reading failed with exception:" << std::endl << e.what() << std::endl;
                }
                cache_status[model_status].push_back(model);
            }
        }
        cache->serialize_cache();
        cache->reset_cache();
    }
    return cache_status;
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
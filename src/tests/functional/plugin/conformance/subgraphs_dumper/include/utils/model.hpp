// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <regex>
#include <unordered_set>
#include <string>
#include <functional>

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
// #ifdef ENABLE_OV_IR_FRONTEND
    std::regex(R"(.*\.xml)"),
// #endif
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

std::pair<std::vector<std::string>, std::pair<ModelCacheStatus, std::vector<std::string>>>
find_models(const std::vector<std::string> &dirs, const std::string& regexp = ".*");

// model_cache_status: model_list
std::map<ModelCacheStatus, std::vector<std::string>> cache_models(
    std::vector<std::shared_ptr<ICache>>& caches,
    const std::vector<std::string>& models,
    bool extract_body);

void save_model_status_to_file(const std::map<ModelCacheStatus, std::vector<std::string>>& caching_status,
                               const std::string& output_dir);

inline ExtractedPattern
generate_model(const std::set<std::shared_ptr<ov::Node>>& nodes,
               std::unordered_set<std::string>& checked_ops,
               const std::string& extractor_name) {
    std::unordered_map<std::string, std::shared_ptr<ov::Node>> model_map;
    // to create reults: { out_op_name, out_ports_without_target_inputs }
    std::unordered_map<std::string, std::unordered_set<size_t>> model_output_nodes;
    std::map<std::string, InputInfo> input_info;
    ov::ParameterVector params;
    {
        // prepare map { original_op_name, cloned_op }
        size_t functional_op_cnt = 0;
        for (const auto& op : nodes) {
            auto op_name = op->get_friendly_name();
            checked_ops.insert(op_name);
            auto cloned_op = clone_node(op, true, false, op->get_friendly_name());
            model_map.insert({ op_name, cloned_op });

            size_t output_cnt = op->outputs().size();
            std::vector<size_t> out_ports(output_cnt);
            std::iota(out_ports.begin(), out_ports.end(), 0);
            std::unordered_set<size_t> out_ports_set(out_ports.begin(), out_ports.end());
            model_output_nodes.insert({ op_name, out_ports_set });
            if (!ov::op::util::is_output(op) && !ov::op::util::is_constant(op) && !ov::op::util::is_parameter(op)) {
                ++functional_op_cnt;
            }
        }

        if (functional_op_cnt < 2) {
            throw std::runtime_error("Incorrect node number to create model");
        }
        // replace new inputs by taken from graph if possible
        for (const auto& op : nodes) {
            int filled_input_idx = -1;
            std::vector<size_t> not_filled_ports;
            auto in_cnt = op->inputs().size();
            auto cloned_op = model_map[op->get_friendly_name()];
            std::map<std::string, InputInfo> this_input_info = get_input_info_by_node(cloned_op);
            for (size_t in_idx = 0; in_idx < in_cnt; ++in_idx) {
                auto in_node = op->get_input_node_ptr(in_idx)->shared_from_this();
                for (size_t in_out_idx = 0; in_out_idx < in_node->outputs().size(); ++in_out_idx) {
                    for (const auto& target_input : in_node->output(in_out_idx).get_target_inputs()) {
                        auto out_in_node = target_input.get_node()->shared_from_this();
                        if (out_in_node == op) {
                            auto in_node_name = in_node->get_friendly_name();
                            auto in_cloned_node = cloned_op->get_input_node_shared_ptr(in_idx);
                            // if op input node is in subgraph
                            if (model_map.count(in_node_name)) {
                                auto in_node = model_map[in_node_name];
                                auto in_cloned_friendly_name = in_cloned_node->get_friendly_name();
                                ov::replace_output_update_name(in_cloned_node->get_default_output(), in_node->output(in_out_idx));
                                in_cloned_node->clear_control_dependencies();
                                if (ov::op::util::is_parameter(in_node)) {
                                    auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(in_node);
                                    params.push_back(param);
                                    this_input_info.insert({ in_node->get_friendly_name(), this_input_info[in_cloned_friendly_name]});
                                } else if (ov::op::util::is_constant(in_node)) {
                                    auto op_to_replace = std::dynamic_pointer_cast<ov::op::v0::Constant>(in_node);
                                    auto param = convert_const_to_param(op_to_replace);
                                    if (param != nullptr) {
                                        params.push_back(param);
                                    }
                                    // insert in_info with updated in_name
                                    this_input_info.insert({ in_node->get_friendly_name(), this_input_info[in_cloned_friendly_name]});
                                }
                                // remove in_info with old name from input info
                                this_input_info.erase(in_cloned_friendly_name);
                                filled_input_idx++;
                                model_output_nodes[in_node_name].erase(in_out_idx);
                                if (model_output_nodes[in_node_name].empty()) {
                                    model_output_nodes.erase(in_node_name);
                                }
                            } else if (ov::op::util::is_parameter(in_cloned_node)) {
                                auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(in_cloned_node);
                                params.push_back(param);
                            } else if (ov::op::util::is_constant(in_cloned_node)) {
                                auto op_to_replace = std::dynamic_pointer_cast<ov::op::v0::Constant>(in_cloned_node);
                                auto param = convert_const_to_param(op_to_replace);
                                if (param != nullptr) {
                                    params.push_back(param);
                                }
                            }
                            break;
                        }
                    }
                    if (filled_input_idx == in_idx) {
                        break;
                    }
                }
            }
            if (!this_input_info.empty()) {
                input_info.insert(this_input_info.begin(), this_input_info.end());
            }
        }
    }
    ov::ResultVector results;
    for (const auto& out_node_name : model_output_nodes) {
        auto out_node = model_map[out_node_name.first];
        if (ov::op::util::is_output(out_node)) {
            results.push_back(std::dynamic_pointer_cast<ov::op::v0::Result>(out_node));
        } else {
            for (const auto& out_port_id : out_node_name.second) {
                results.push_back(std::make_shared<ov::op::v0::Result>(out_node->output(out_port_id)));
            }
        }
    }
    auto model = std::make_shared<ov::Model>(results, params);
    std::string string_to_hash;
    for (const auto& op : model->get_ordered_ops()) {
        std::ostringstream result;
        result << op->get_type_info();
        for (const auto& in : op->inputs()) { 
            result << in.get_element_type();
            result << in.get_partial_shape().rank();
            result << in.get_partial_shape().is_static();
        }
        for (const auto& out : op->outputs()) {
            result << out.get_element_type();
            result << out.get_partial_shape().rank();
            result << out.get_partial_shape().is_static();
        }
        string_to_hash += result.str();
    }
    for (const auto& in : input_info) {
        string_to_hash += (in.second.is_const ? "1" : "0");
    }
    auto h1 = std::hash<std::string>{}(string_to_hash);
    model->set_friendly_name(std::to_string(h1));
    return { model, input_info, extractor_name };
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov

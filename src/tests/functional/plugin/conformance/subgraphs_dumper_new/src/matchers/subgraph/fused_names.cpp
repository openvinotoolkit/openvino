// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/op/util/op_types.hpp"

#include "functional_test_utils/ov_plugin_cache.hpp"

#include "matchers/subgraph/fused_names.hpp"
#include "utils/node.hpp"

using namespace ov::tools::subgraph_dumper;

std::list<BaseMatcher::ExtractedPattern>
FusedNamesMatcher::extract(const std::shared_ptr<ov::Model> &model) {
    std::list<BaseMatcher::ExtractedPattern> matched_patterns;
    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(model);
    bool is_graph_started = false;
    std::map<std::string, std::shared_ptr<ov::Node>> model_map;
    std::unordered_set<std::string> compiled_op_name;
    for (const auto& compiled_op : compiled_model.get_runtime_model()->get_ordered_ops()) {
        const auto& rt_info = compiled_op->get_rt_info();
        if (rt_info.count("originalLayersNames")) {
            compiled_op_name.insert(rt_info.find("originalLayersNames")->second.as<std::string>());
        }
    }

    for (const auto& op : model->get_ordered_ops()) {
        auto op_name = op->get_friendly_name();
        if (ov::op::util::is_parameter(op) || ov::op::util::is_constant(op)) {
            continue;
        }
        auto cloned_op = clone_node(op, true, false, "Op_" + std::to_string(model_map.size()));
        if (model_map.empty()) {
            model_map.insert({ op->get_friendly_name(), cloned_op });
        } else {
            size_t inputs_size = op->inputs().size();
            ov::OutputVector in_out_vector(inputs_size);
            for (size_t in_idx = 0; in_idx < inputs_size; ++in_idx) {
                auto in_node = op->get_input_node_ptr(in_idx)->shared_from_this();
                for (size_t in_out_idx = 0; in_out_idx < in_node->outputs().size(); ++in_out_idx) {
                    bool is_input_filled = false;
                    for (const auto& target_input : in_node->output(in_out_idx).get_target_inputs()) {
                        auto out_in_node = target_input.get_node()->shared_from_this();
                        if (out_in_node == op) {
                            auto in_node_name = in_node->get_friendly_name();
                            in_out_vector[in_idx] = model_map.count(in_node_name) ?
                                               model_map.at(in_node_name)->output(in_out_idx) :
                                               cloned_op->get_input_node_ptr(in_idx)->output(in_out_idx);
                            is_input_filled = true;
                            break;
                        }
                    }
                    if (is_input_filled) {
                        break;
                    }
                }
            }
            model_map.insert({ op_name, cloned_op->clone_with_new_inputs(in_out_vector) });
        }
        if (!compiled_op_name.count(op_name) || ov::op::util::is_output(op)) {
            if (model_map.size() > 1) {
                ov::OutputVector results;
                std::map<std::string, InputInfo> input_info;
                for (const auto& op : model_map) {
                    auto this_input_info = get_input_info_by_node(op.second);
                    input_info.insert(this_input_info.begin(), this_input_info.end());
                    for (size_t j = 0; j < op.second->outputs().size(); ++j) {
                        if (op.second->output(j).get_target_inputs().empty()) {
                            results.push_back(std::make_shared<ov::op::v0::Result>(op.second->output(j)));
                        }
                    }
                }
                matched_patterns.push_back({std::make_shared<ov::Model>(results), input_info});
            }
            model_map.clear();
        }
    }
    return matched_patterns;
}


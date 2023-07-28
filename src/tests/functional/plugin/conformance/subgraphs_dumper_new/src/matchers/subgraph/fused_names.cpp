// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"

#include "functional_test_utils/ov_plugin_cache.hpp"

#include "matchers/subgraph/fused_names.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

std::unordered_set<std::string>
FusedNamesExtractor::extract_compiled_model_names(const std::shared_ptr<ov::Model>& model) {
    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(model);
    std::unordered_set<std::string> compiled_op_name;
    for (const auto& compiled_op : compiled_model.get_runtime_model()->get_ordered_ops()) {
        const auto& rt_info = compiled_op->get_rt_info();
        if (rt_info.count("originalLayersNames")) {
            compiled_op_name.insert(rt_info.find("originalLayersNames")->second.as<std::string>());
        }
    }
    return compiled_op_name;
}

std::list<ExtractedPattern>
FusedNamesExtractor::extract(const std::shared_ptr<ov::Model> &model,
                             bool is_extract_body) {
    auto compiled_op_name = extract_compiled_model_names(model);
    std::list<ExtractedPattern> matched_patterns;
    std::unordered_set<std::string> checked_ops;
    std::set<std::shared_ptr<ov::Node>> nodes;
    std::shared_ptr<ov::Node> start_node = nullptr;
    for (const auto& op : model->get_ordered_ops()) {
        auto op_name = op->get_friendly_name();
        if (is_node_to_skip(op) || checked_ops.count(op_name)) {
            continue;
        }
        if (start_node == nullptr) {
            start_node = op;
        }
        nodes.insert(op);
        if (is_extract_body) {
            if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
                auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                auto tmp_res = extract(ti_body);
                matched_patterns.insert(matched_patterns.end(), tmp_res.begin(), tmp_res.end());
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                auto tmp_res = extract(loop_body);
                matched_patterns.insert(matched_patterns.end(), tmp_res.begin(), tmp_res.end());
            } else if (std::dynamic_pointer_cast<ov::op::v8::If>(op)) {
                auto if_op = ov::as_type_ptr<ov::op::v8::If>(op);
                std::vector<std::shared_ptr<ov::Model>> bodies;
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    auto tmp_res = extract(if_body);
                    matched_patterns.insert(matched_patterns.end(), tmp_res.begin(), tmp_res.end());
                }
            }
        }
        if (!compiled_op_name.count(op_name)) {
            try {
                matched_patterns.push_back(generate_model(nodes, start_node, checked_ops));
            } catch(std::exception& e) {
                std::cout << "[ ERROR ] Impossible to generate network and add to GraphCache: " <<e.what() << std::endl;
            }
            start_node = nullptr;
            nodes.clear();
        }
    }
    try {
        matched_patterns.push_back(generate_model(nodes, start_node, checked_ops));
    } catch(std::exception& e) {
        std::cout << "[ ERROR ] Impossible to generate network and add to GraphCache: " << e.what() << std::endl;
    }
    return matched_patterns;
}

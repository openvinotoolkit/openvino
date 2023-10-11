// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <vector>

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"

#include "matchers/subgraph/repeat_pattern.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

std::list<ExtractedPattern>
RepeatPatternExtractor::extract(const std::shared_ptr<ov::Model> &model,
                                bool is_extract_body,
                                bool is_copy_constants) {
    std::unordered_set<std::string> checked_ops;
    std::list<ExtractedPattern> to_cache;

    auto ordered_ops = model->get_ordered_ops();
    auto op_cnt = ordered_ops.size();

    for (size_t idx = 0; idx < op_cnt; ++idx) {
        auto op = ordered_ops[idx];
        auto op_name = op->get_friendly_name();
        if (checked_ops.count(op_name)|| is_node_to_skip(op)) {
            continue;
        }

        std::vector<size_t> start_node_idx{idx};
        for (size_t i = idx + 1; i < op_cnt; ++i) {
            if (manager.match(op, ordered_ops[i])) {
                start_node_idx.push_back(i);
            }
        }
        if (start_node_idx.size() < 2) {
            checked_ops.insert(op_name);
            continue;
        }

        std::vector<std::set<std::shared_ptr<ov::Node>>> nodes(start_node_idx.size());
        for (size_t i = 0; i < start_node_idx.size(); ++i) {
            for (size_t j = i + 1; j < start_node_idx.size(); ++j) {
                size_t node_idx = start_node_idx[i], ref_node_idx = start_node_idx[j];
                while (node_idx < op_cnt && ref_node_idx < op_cnt) {
                    auto node = ordered_ops[node_idx];
                    auto ref_node = ordered_ops[ref_node_idx];
                    if (checked_ops.count(node->get_friendly_name()) ||
                        checked_ops.count(ref_node->get_friendly_name())) {
                        break;
                    }
                    if (!is_node_to_skip(node) && !is_node_to_skip(ref_node)) {
                        if (node_idx == start_node_idx[i] && ref_node_idx == start_node_idx[j]) {
                                nodes[i].insert(node);
                                nodes[j].insert(ref_node);
                        } else if (manager.match(node, ref_node)) {
                            // check if we met the same node
                            if (manager.match(node, op)) {
                                break;
                            }
                            if (checked_ops.count(node->get_friendly_name()) ||
                                checked_ops.count(ref_node->get_friendly_name())) {
                                break;
                            }
                            // check that any input node is using in graph
                            bool is_input_in_graph = false;
                            for (size_t in_idx = 0; in_idx < node->inputs().size(); ++in_idx) {
                                auto in_node = node->get_input_node_ptr(in_idx)->shared_from_this();
                                auto ref_in_node = ref_node->get_input_node_ptr(in_idx)->shared_from_this();
                                if (nodes[i].count(in_node) && nodes[j].count(ref_in_node)) {
                                    is_input_in_graph = true;
                                    break;
                                }
                            }
                            if (!is_input_in_graph) {
                                break;
                            }

                            nodes[i].insert(ordered_ops[node_idx]);
                            nodes[j].insert(ordered_ops[ref_node_idx]);
                        } else {
                            break;
                        }
                    }
                    ++node_idx;
                    ++ref_node_idx;
                }
            }
        }
        for (size_t i = 0; i < start_node_idx.size(); ++i) {
            try {
                std::unordered_set<std::string> tmp_checked_ops;
                auto extracted_pattern = generate_model(nodes[i], tmp_checked_ops, extractor_name, is_copy_constants);
                auto extracted_model = std::get<0>(extracted_pattern);
                std::list<ExtractedPattern> secondary_patterns;
                if (nodes[i].size() > 20) {
                    secondary_patterns = extract(std::get<0>(extracted_pattern), is_extract_body, is_copy_constants);
                }
                if (secondary_patterns.size() > 1) {
                    to_cache.insert(to_cache.end(), secondary_patterns.begin(), secondary_patterns.end());
                } else {
                    to_cache.push_back(extracted_pattern);
                }
                nodes[i].clear();
                checked_ops.insert(tmp_checked_ops.begin(), tmp_checked_ops.end());
            } catch(std::exception& e) {
                if (std::string(e.what()).find("Incorrect node number to create model!") == std::string::npos) {
                    // std::cout << "[ WARNING ] Impossible to generate network and add to GraphCache: " <<e.what() << std::endl;
                }
            }
        }
        if (is_extract_body) {
            if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
                auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                auto tmp_res = extract(ti_body);
                to_cache.insert(to_cache.end(), tmp_res.begin(), tmp_res.end());
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                auto tmp_res = extract(loop_body);
                to_cache.insert(to_cache.end(), tmp_res.begin(), tmp_res.end());
            } else if (std::dynamic_pointer_cast<ov::op::v8::If>(op)) {
                auto if_op = ov::as_type_ptr<ov::op::v8::If>(op);
                std::vector<std::shared_ptr<ov::Model>> bodies;
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    auto tmp_res = extract(if_body);
                    to_cache.insert(to_cache.end(), tmp_res.begin(), tmp_res.end());
                }
            }
        }
    }
    return to_cache;
}

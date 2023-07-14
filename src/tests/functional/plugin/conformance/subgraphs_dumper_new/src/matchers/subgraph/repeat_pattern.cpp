// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <vector>

#include "matchers/subgraph/repeat_pattern.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

std::list<ExtractedPattern>
RepeatPatternExtractor::extract(const std::shared_ptr<ov::Model> &model,
                                bool is_extract_body) {
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
        std::set<std::shared_ptr<ov::Node>> unique_nodes;
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
                                unique_nodes.insert(node < ref_node ? node : ref_node);
                        } else if (manager.match(node, ref_node)) {
                            // check if we met the same node
                            if (manager.match(node, op)) {
                                break;
                            }
                            bool is_met_before = false;
                            for (const auto& unique_node : unique_nodes) {
                                if (manager.match(node, unique_node)) {
                                    is_met_before = true;
                                    break;
                                }
                            }
                            if (is_met_before) {
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
                            unique_nodes.insert(node < ref_node ? node : ref_node);
                        }
                    }
                    ++node_idx;
                    ++ref_node_idx;
                }
            }
        }
        for (size_t i = 0; i < start_node_idx.size(); ++i) {
            try {
                to_cache.push_back(
                    generate_model(nodes[i], ordered_ops[start_node_idx[i]], checked_ops));
                nodes[i].clear();
            } catch(std::exception& e) {
                std::cout << e.what() << std::endl;
            }
        }
    }
    return to_cache;
}

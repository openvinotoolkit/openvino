// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <vector>

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"

#include "matchers/subgraph/repeat_pattern.hpp"
#include "utils/model.hpp"
#include "utils/model_comparator.hpp"

using namespace ov::tools::subgraph_dumper;

void RepeatPatternExtractor::set_recursive_extraction(bool _is_recursive_extraction) {
    is_recursive_extraction = _is_recursive_extraction;
}

std::vector<ExtractedPattern>
RepeatPatternExtractor::extract(const std::shared_ptr<ov::Model> &model) {
    std::vector<ExtractedPattern> extracted_patterns;
    for (const auto& pattern : find_repeat_patterns(model)) {
        for (const auto& pattern_structure : pattern) {
            extracted_patterns.push_back({std::get<0>(pattern_structure), std::get<2>(pattern_structure), extractor_name});
        }
    }
    return extracted_patterns;
}

std::vector<std::vector<RepeatPatternExtractor::PatternBorders>>
RepeatPatternExtractor::get_repeat_pattern_borders(const std::shared_ptr<ov::Model> &model) {
    std::vector<std::vector<RepeatPatternExtractor::PatternBorders>> extracted_patterns;
    for (auto& pattern : find_repeat_patterns(model, true)) {
        std::vector<RepeatPatternExtractor::PatternBorders> same_pattern_borders;
        for (const auto& pattern_structure : pattern) {
            std::set<std::string> output_names;
            for (const auto& result : std::get<0>(pattern_structure)->get_results()) {
                output_names.insert(result->get_input_node_shared_ptr(0)->get_friendly_name());
            }

            RepeatPatternExtractor::InputVector in_vec;
            RepeatPatternExtractor::OutputVector out_vec;
            for (const auto& node : std::get<1>(pattern_structure)) {
                if (output_names.count(node->get_friendly_name())) {
                    OutputVector node_outputs = node->outputs();
                    out_vec.insert(out_vec.end(), node_outputs.begin(), node_outputs.end());
                } else {
                    for (const auto& input : node->inputs()) {
                        in_vec.push_back(input);
                    }
                }
            }
            same_pattern_borders.push_back({in_vec, out_vec});
        }
        extracted_patterns.push_back(same_pattern_borders);
    }
    return extracted_patterns;
}

std::vector<std::vector<ov::NodeVector>>
RepeatPatternExtractor::get_repeat_node_vectors(const std::shared_ptr<ov::Model> &model) {
    std::vector<std::vector<ov::NodeVector>> extracted_patterns;
    for (const auto& pattern : find_repeat_patterns(model)) {
        std::vector<ov::NodeVector> same_pattern_nodes;
        for (const auto& pattern_structure : pattern) {
            same_pattern_nodes.push_back(std::get<1>(pattern_structure));
        }
        extracted_patterns.push_back(same_pattern_nodes);
    }
    return extracted_patterns;
}

void
RepeatPatternExtractor::update_extractor_cache(
    std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>>& extracted_patterns,
    const std::shared_ptr<ov::Model>& pattern,
    const ov::NodeVector& pattern_node_vector,
    const std::map<std::string, InputInfo>& pattern_in_info) {
    for (auto& extracted_pattern : extracted_patterns) {
        auto& pattern_structure = extracted_pattern.front();
        const auto& cached_pattern = std::get<0>(pattern_structure);
        if (model_comparator->match(pattern, cached_pattern)) {
            try {
                const auto& cached_in_info = std::get<2>(pattern_structure);
                align_input_info(pattern, cached_pattern, pattern_in_info, cached_in_info);
                extracted_pattern.push_back({ pattern, pattern_node_vector, pattern_in_info });
                return;
            } catch(std::exception) {}
        }
    }
    extracted_patterns.push_back({{ pattern, pattern_node_vector, pattern_in_info }});
}

void
RepeatPatternExtractor::update_extractor_cache(
    std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>>& extracted_patterns,
    std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>>& secondary_extracted_patterns) {
    auto extern_it = secondary_extracted_patterns.begin();
    while (!secondary_extracted_patterns.empty()) {
        auto it = extern_it->rbegin();
        while (!extern_it->empty()) {
            auto& pattern_structure = *it;
            const auto& pattern = std::get<0>(pattern_structure);
            const auto& pattern_node_vector = std::get<1>(pattern_structure);
            const auto& pattern_in_info = std::get<2>(pattern_structure);
            update_extractor_cache(extracted_patterns, pattern, pattern_node_vector, pattern_in_info);
            extern_it->pop_back();
            it = extern_it->rbegin();
        }
        secondary_extracted_patterns.pop_front();
    }
}

std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>>
RepeatPatternExtractor::find_repeat_patterns(const std::shared_ptr<ov::Model> &model,
                                             bool is_save_borders_only) {
    std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>> extracted_patterns;
    std::unordered_set<std::string> checked_ops;

    auto ordered_ops = model->get_ordered_ops();
    auto op_cnt = ordered_ops.size();

    for (size_t idx = 0; idx < op_cnt; ++idx) {
        auto op = ordered_ops[idx];
        auto op_name = op->get_friendly_name();
        if (checked_ops.count(op_name)|| is_node_to_skip(op)) {
            continue;
        }

        // find the same nodes
        std::vector<size_t> start_node_idx{idx};
        for (size_t i = idx + 1; i < op_cnt; ++i) {
            if (model_comparator->match(op, ordered_ops[i])) {
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
                        } else if (model_comparator->match(node, ref_node)) {
                            // check if we met the same node
                            if (model_comparator->match(node, op)) {
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
                // model, in_info, extractor_name
                ov::NodeVector nodes_vector(nodes[i].begin(), nodes[i].end());
                auto extracted_pattern = generate_model(nodes_vector, tmp_checked_ops, is_save_const, is_save_borders_only);
                auto extracted_model = extracted_pattern.first;
                if (is_recursive_extraction && nodes_vector.size() > 20) {
                    auto secondary_patterns = find_repeat_patterns(extracted_model, is_save_borders_only);
                    if (!secondary_patterns.empty()) {
                        tmp_checked_ops.clear();
                        update_extractor_cache(extracted_patterns, secondary_patterns);
                    } else {
                        update_extractor_cache(extracted_patterns,
                                               extracted_model,
                                               nodes_vector,
                                               extracted_pattern.second);
                    }
                } else {
                    update_extractor_cache(extracted_patterns,
                                           extracted_model,
                                           nodes_vector,
                                           extracted_pattern.second);
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
                auto secondary_patterns = find_repeat_patterns(ti_body, is_save_borders_only);
                update_extractor_cache(extracted_patterns, secondary_patterns);
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                auto secondary_patterns = find_repeat_patterns(loop_body, is_save_borders_only);
                update_extractor_cache(extracted_patterns, secondary_patterns);
            } else if (std::dynamic_pointer_cast<ov::op::v8::If>(op)) {
                auto if_op = ov::as_type_ptr<ov::op::v8::If>(op);
                std::vector<std::shared_ptr<ov::Model>> bodies;
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    auto secondary_patterns = find_repeat_patterns(if_body, is_save_borders_only);
                    update_extractor_cache(extracted_patterns, secondary_patterns);
                }
            }
        }
    }

    // clean up patterns
    {
        auto it = extracted_patterns.begin();
        size_t elem_cnt = 0;
        while (it != extracted_patterns.end()) {
            if (it->size() > 1) {
                ++it;
                ++elem_cnt;
            } else {
                extracted_patterns.erase(it);
                it = extracted_patterns.begin();
                std::advance(it, elem_cnt);
            }
        }
    }
    return extracted_patterns;
}

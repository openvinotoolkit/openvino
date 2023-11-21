// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "matchers/subgraph/repeat_pattern.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

void RepeatPatternExtractor::set_recursive_extraction(bool _is_recursive_extraction) {
    is_recursive_extraction = _is_recursive_extraction;
}

std::vector<RepeatPatternExtractor::ExtractedPattern>
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
    const std::map<std::string, ov::conformance::InputInfo>& pattern_in_info) {
    for (auto& extracted_pattern : extracted_patterns) {
        auto& pattern_structure = extracted_pattern.front();
        const auto cached_pattern = std::get<0>(pattern_structure);
        if (model_comparator->match(pattern, cached_pattern)) {
            try {
                const auto& cached_in_info = std::get<2>(pattern_structure);
                ov::util::align_input_info(pattern, cached_pattern, pattern_in_info, cached_in_info);
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
    while (!secondary_extracted_patterns.empty()) {
        auto extern_it = secondary_extracted_patterns.begin();
        while (!extern_it->empty()) {
            auto& pattern_structure = *(extern_it->rbegin());
            const auto& pattern = std::get<0>(pattern_structure);
            const auto& pattern_node_vector = std::get<1>(pattern_structure);
            const auto& pattern_in_info = std::get<2>(pattern_structure);
            update_extractor_cache(extracted_patterns, pattern, pattern_node_vector, pattern_in_info);
            extern_it->pop_back();
        }
        secondary_extracted_patterns.pop_front();
    }
}

std::vector<ov::NodeVector>
RepeatPatternExtractor::get_node_vector(const ov::NodeVector& start_node_vec) {
    using NodePair = std::pair<std::shared_ptr<ov::Node>, std::vector<size_t>>;
    auto pattern_cnt = start_node_vec.size();
    std::vector<std::vector<NodePair>> patterns(pattern_cnt);
    std::unordered_set<std::string> checked_ops;

    size_t max_node_cnt = 0;
    for (size_t pattern_idx = 0; pattern_idx < pattern_cnt; ++pattern_idx) {
        if (!patterns[pattern_idx].empty()) {
            continue;
        }

        std::list<std::shared_ptr<ov::Node>> queue = {start_node_vec[pattern_idx]};
        // std::set<std::shared_ptr<ov::Node>> side_queue;
        while (!queue.empty()) {
            const auto node = queue.front();

            // if (checked_ops.count(node->get_friendly_name())) {
            //     queue.pop_front();
            //     continue;
            // }
            bool is_all_in_checked = true;
            for (size_t node_vec_idx = pattern_idx; node_vec_idx < pattern_cnt; ++node_vec_idx) {
                auto& node_vector = patterns[node_vec_idx];
                if (node_vector.empty()) {
                    if (node == start_node_vec[node_vec_idx]) {
                        node_vector.push_back({node, {}});
                    }
                    break;
                }
                std::vector<size_t> input_indexes;
                size_t first_in_idx = node_vector.size();
                for (size_t in_idx = 0; in_idx < node->inputs().size(); in_idx++) {
                    auto input_node = node->get_input_node_shared_ptr(in_idx);
                    if (!checked_ops.count(input_node->get_friendly_name())) {
                        // side_queue.insert(node);
                        is_all_in_checked = true;
                        break;
                    }
                    if (input_indexes.empty()) {
                        while (first_in_idx > 0) {
                            --first_in_idx;
                            if (node_vector[first_in_idx].first == input_node) {
                                input_indexes.push_back(first_in_idx);
                                break;
                            }
                        }
                    } else {
                        for (size_t i = first_in_idx; i < node_vector.size(); ++i) {
                            if (node_vector[i].first == input_node) {
                                input_indexes.push_back(i);
                                first_in_idx = i;
                                break;
                            }
                        }
                    }
                }
                if (input_indexes.empty()) {
                    node_vector.push_back({node, input_indexes});
                }
                if (node_vector.size() > max_node_cnt) {
                    max_node_cnt = node_vector.size();
                }
                if (is_all_in_checked) {
                    break;
                }
            }

            if (node->get_friendly_name() == std::string("BatchNormalization_18")) {
                auto a = 0;
            }

            for (size_t out_idx = 0; out_idx < node->get_output_size(); ++out_idx) {
                for (const auto& target_input : node->get_output_target_inputs(out_idx)) {
                    auto output_node = target_input.get_node()->shared_from_this();
                    bool is_output_node = ov::op::util::is_output(output_node);
                    if (!is_output_node) {
                        queue.push_back(output_node);
                        // if (side_queue.count(output_node)) {
                            // side_queue.erase(output_node);
                        // }
                    }
                }
            }
            if (!is_all_in_checked)
                checked_ops.insert(node->get_friendly_name());
            queue.pop_front();
        }
    }
    std::cout << patterns.size() << " " << max_node_cnt << std::endl;
    for (auto& node_vec : patterns) {
        size_t dist = max_node_cnt - node_vec.size();
        std::cout << dist << std::endl;
        node_vec.insert(node_vec.end(), dist, {nullptr, {}});
    }

    for (size_t j = 0; j < max_node_cnt; ++j) {
        std::vector<bool> mask(pattern_cnt, false);
        for (size_t i_orig = 0; i_orig < pattern_cnt; ++i_orig) {
            if (patterns[i_orig][j].first == nullptr) {
                continue;
            }

            bool is_not_empty_in = patterns[i_orig][j].second.empty();
            std::vector<size_t> updated_inputs;
            for (const auto& in_idx : patterns[i_orig][j].second) {
                if (patterns[i_orig][in_idx].first != nullptr) {
                    is_not_empty_in = true;
                    updated_inputs.push_back(in_idx);
                }
            }
            if (!is_not_empty_in) {
                patterns[i_orig][j].first = nullptr;
                continue;
            }
            patterns[i_orig][j].second = updated_inputs;

            for (size_t i_ref = i_orig + 1; i_ref < patterns.size(); ++i_ref) {
                if (mask[i_orig] && mask[i_ref]) {
                    continue;
                }

                if (patterns[i_ref][j].first == nullptr) {
                    continue;
                }

                if (model_comparator->match(patterns[i_ref][j].first, patterns[i_orig][j].first)) {
                    mask[i_orig] = true;
                    mask[i_ref] = true;
                }
            }
        }

        for (size_t i = 0; i < pattern_cnt; ++i) {
            if (!mask[i]) {
                patterns[i][j].first = nullptr;
            }
        }
    }

    std::vector<ov::NodeVector> result;
    for (const auto& node_vec : patterns) {
        ov::NodeVector tmp;
        for (const auto& node_item : node_vec) {
            if (node_item == node_vec.front() || node_item.first != nullptr && !node_item.second.empty()) {
                tmp.push_back(node_item.first);
            }
        }
        if (tmp.size() > 1) {
            result.push_back(tmp);
        }
    }
    return result;
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
        if (checked_ops.count(op_name)|| ov::util::is_node_to_skip(op)) {
            continue;
        }

        // find the same nodes
        // std::vector<size_t> pattern_idx{idx};
        ov::NodeVector pattern_start_nodes;
        for (size_t i = idx + 1; i < op_cnt; ++i) {
            if (model_comparator->match(op, ordered_ops[i])) {
                // pattern_idx.push_back(i);
                pattern_start_nodes.push_back(ordered_ops[i]);
            }
        }
        // if (pattern_idx.size() < 2) {
        if (pattern_start_nodes.size() < 2) {
            checked_ops.insert(op_name);
            continue;
        }

        // std::vector<std::set<std::shared_ptr<ov::Node>>> nodes(pattern_idx.size());
        // for (size_t i = 0; i < pattern_idx.size(); ++i) {
        //     for (size_t j = i + 1; j < pattern_idx.size(); ++j) {
        //         size_t node_idx = pattern_idx[i], ref_node_idx = pattern_idx[j];
        //         while (node_idx < op_cnt && ref_node_idx < op_cnt) {
        //             auto node = ordered_ops[node_idx];
        //             auto ref_node = ordered_ops[ref_node_idx];
        //             if (checked_ops.count(node->get_friendly_name()) ||
        //                 checked_ops.count(ref_node->get_friendly_name())) {
        //                 break;
        //             }
        //             if (!ov::util::is_node_to_skip(node) &&
        //                 !ov::util::is_node_to_skip(ref_node)) {
        //                 if (node_idx == pattern_idx[i] && ref_node_idx == pattern_idx[j]) {
        //                         nodes[i].insert(node);
        //                         nodes[j].insert(ref_node);
        //                 } else if (model_comparator->match(node, ref_node)) {
        //                     // check if we met the same node
        //                     if (model_comparator->match(node, op)) {
        //                         break;
        //                     }
        //                     if (checked_ops.count(node->get_friendly_name()) ||
        //                         checked_ops.count(ref_node->get_friendly_name())) {
        //                         break;
        //                     }
        //                     // check that any input node is using in graph
        //                     bool is_input_in_graph = false;
        //                     for (size_t in_idx = 0; in_idx < node->inputs().size(); ++in_idx) {
        //                         auto in_node = node->get_input_node_ptr(in_idx)->shared_from_this();
        //                         auto ref_in_node = ref_node->get_input_node_ptr(in_idx)->shared_from_this();
        //                         if (nodes[i].count(in_node) && nodes[j].count(ref_in_node)) {
        //                             is_input_in_graph = true;
        //                             break;
        //                         }
        //                     }
        //                     if (!is_input_in_graph) {
        //                         break;
        //                     }

        //                     nodes[i].insert(ordered_ops[node_idx]);
        //                     nodes[j].insert(ordered_ops[ref_node_idx]);
        //                 } else {
        //                     break;
        //                 }
        //             }
        //             ++node_idx;
        //             ++ref_node_idx;
        //         }
        //     }
        // }
        auto potential_patterns = get_node_vector(pattern_start_nodes);
        // for (size_t i = 0; i < pattern_idx.size(); ++i) {
        for (auto& nodes_vector : potential_patterns) {
            try {
                std::unordered_set<std::string> tmp_checked_ops;
                // model, in_info, extractor_name
                // ov::NodeVector nodes_vector(nodes[i].begin(), nodes[i].end());
                // auto extracted_pattern = ov::util::generate_model(nodes_vector, tmp_checked_ops, is_save_const, is_save_borders_only);
                auto extracted_pattern = ov::util::generate_model(nodes_vector, tmp_checked_ops, is_save_const, is_save_borders_only);
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
                // nodes[i].clear();
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

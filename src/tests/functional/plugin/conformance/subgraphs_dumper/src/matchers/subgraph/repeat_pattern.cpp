// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "matchers/subgraph/repeat_pattern.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

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
    auto a = find_repeat_patterns(model, true);
    for (auto& pattern : a) {
        std::vector<RepeatPatternExtractor::PatternBorders> same_pattern_borders;
        for (const auto& pattern_structure : pattern) {
            RepeatPatternExtractor::InputVector in_vec;
            RepeatPatternExtractor::OutputVector out_vec;
            auto node_vector = std::get<1>(pattern_structure);
            for (const auto& node : node_vector) {
                for (size_t out_idx = 0; out_idx < node->outputs().size(); ++out_idx) {
                    size_t idx = 0;
                    const auto target_inputs = node->get_output_target_inputs(out_idx);
                    for (const auto& target_input :  target_inputs) {
                        const auto target_in_node = target_input.get_node()->shared_from_this();
                        if (std::find(node_vector.begin(), node_vector.end(), target_in_node) == node_vector.end()) {
                            ++idx;
                        }
                    }
                    if (idx == target_inputs.size()) {
                        out_vec.push_back(node->output(out_idx));
                    }
                }
                for (size_t in_idx = 0; in_idx < node->inputs().size(); ++in_idx) {
                    auto in_node = node->get_input_node_shared_ptr(in_idx);
                    if (std::find(node_vector.begin(), node_vector.end(), in_node) == node_vector.end()) {
                        in_vec.push_back(node->input(in_idx));
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
    const std::vector<ov::NodeVector>& pattern_node_vector,
    const std::map<std::string, ov::conformance::InputInfo>& pattern_in_info) {
    for (auto& extracted_pattern : extracted_patterns) {
        auto& pattern_structure = extracted_pattern.front();
        const auto cached_pattern = std::get<0>(pattern_structure);
        if (model_comparator->match(pattern, cached_pattern)) {
            try {
                const auto& cached_in_info = std::get<2>(pattern_structure);
                ov::util::align_input_info(pattern, cached_pattern,
                                           pattern_in_info, cached_in_info,
                                           model_comparator->get_matched_ops_in_graphs(pattern, cached_pattern));
                for (const auto& p : pattern_node_vector) {
                    extracted_pattern.push_back({ pattern, p, pattern_in_info });
                }
                return;
            } catch(std::exception) {}
        }
    }
    extracted_patterns.push_back({{ pattern, pattern_node_vector.front(), pattern_in_info }});
    for (size_t i = 1; i < pattern_node_vector.size(); ++i) {
        extracted_patterns.back().push_back({ pattern, pattern_node_vector[i], pattern_in_info });
    }
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
            update_extractor_cache(extracted_patterns, pattern, {pattern_node_vector}, pattern_in_info);
            extern_it->pop_back();
        }
        secondary_extracted_patterns.pop_front();
    }
}


std::vector<std::vector<ov::NodeVector>>
RepeatPatternExtractor::get_patterns_by_nodes(const std::vector<size_t>& start_op_vec,
                                              const ov::NodeVector& ordered_ops) {
    // handle case impossible to extract repeat patterns
    if (start_op_vec.size() < 2 || ordered_ops.size() < 3) {
        return {{}};
    }
    // is_recursive_extraction = true;
    // prepare node vectors contains potential patterns from start_node to output
    // first one is biggest subgraph, last one is smallest one
    auto pattern_cnt = start_op_vec.size();
    std::vector<ov::NodeVector> patterns(pattern_cnt);
    for (size_t pattern_idx = 0; pattern_idx < pattern_cnt; ++pattern_idx) {
        // get only nodes are after start node in graph
        std::unordered_set<std::shared_ptr<ov::Node>> nodes_to_check;
        const auto& start_op_idx = start_op_vec[pattern_idx];
        util::get_subgraph_set_node(nodes_to_check, ordered_ops[start_op_idx]);
        for (const auto& op : ordered_ops) {
            if (nodes_to_check.count(op)) {
                patterns[pattern_idx].push_back(op);
            }
        }
    }
    {
        // reverse node vectors to anylize from small to big subgraphs
        std::reverse(patterns.begin(), patterns.end());
        std::vector<ov::NodeVector> potential_patterns(pattern_cnt);
        for (size_t i_orig = 0; i_orig < pattern_cnt - 1; ++i_orig) {
            // skip comparation in case pattern is iniatized
            if (!potential_patterns[i_orig].empty()) {
                continue;
            }
            for (size_t i_ref = i_orig + 1; i_ref < pattern_cnt; ++i_ref) {
                if (!potential_patterns[i_ref].empty()) {
                    continue;
                }
                // extract minimal intersected patterns
                auto intersection_len = std::min(patterns[i_orig].size(), patterns[i_ref].size());
                ov::NodeVector pattern_orig(intersection_len, nullptr), pattern_ref(intersection_len, nullptr);
                for (size_t j = 0; j < intersection_len; ++j) {
                    if (model_comparator->match(patterns[i_orig][j], patterns[i_ref][j])) {
                        if (patterns[i_orig][j] == patterns[i_ref][j]) {
                            break;
                        }
                        if (is_split_by_matched_nodes) {
                            if (model_comparator->match(patterns[i_orig][0], patterns[i_orig][j]) ||
                                model_comparator->match(patterns[i_ref][0], patterns[i_ref][j])) {
                                break;
                            }
                        }
                        // check inputs and matching in case not start_node
                        if (j != 0) {
                            bool is_input_matched = false;
                            for (size_t input_idx = 0; input_idx < patterns[i_orig][j]->inputs().size(); ++input_idx) {
                                auto in_orig = patterns[i_orig][j]->get_input_node_shared_ptr(input_idx);
                                auto in_ref = patterns[i_ref][j]->get_input_node_shared_ptr(input_idx);
                                if (std::find(pattern_orig.begin(), pattern_orig.end(), in_orig) != pattern_orig.end() &&
                                    std::find(pattern_ref.begin(), pattern_ref.end(), in_ref) != pattern_ref.end()) {
                                    is_input_matched = true;
                                    break;
                                }
                            }
                            if (!is_input_matched) {
                                continue;
                            }
                        }
                        pattern_orig[j] = patterns[i_orig][j];
                        pattern_ref[j] = patterns[i_ref][j];
                    }
                }
                // fill vectors only by valid nodes
                ov::NodeVector orig, ref;
                for (size_t node_idx = 0; node_idx < pattern_orig.size(); ++node_idx) {
                    if (pattern_orig[node_idx] != 0) {
                        orig.emplace_back(pattern_orig[node_idx]);
                    }
                    if (pattern_ref[node_idx] != 0) {
                        ref.emplace_back(pattern_ref[node_idx]);
                    }
                }
                if (orig.size() < min_graph_size) {
                    continue;
                }
                potential_patterns[i_orig] = orig;
                potential_patterns[i_ref] = ref;
            }
        }
        // sort patterns by node vectors size
        std::sort(potential_patterns.begin(), potential_patterns.end(), [](const ov::NodeVector& a, const ov::NodeVector& b) {
            return a.size() > b.size();
        });

        // exclude not repeated pattern
        while (!potential_patterns.empty() && potential_patterns.rbegin()->size() < 2) {
            potential_patterns.pop_back();
        }
        patterns = potential_patterns;
    }

    // group node vectors to the patterns: std::vector<ov::NodeVector>
    std::vector<std::vector<ov::NodeVector>> pattern_vec;
    for (size_t pattern_idx = 0; pattern_idx < patterns.size(); ++pattern_idx) {
        const auto& pattern = patterns[pattern_idx];
        if (pattern_vec.empty()) {
            pattern_vec.push_back({{pattern}});
        } else if (pattern_vec.rbegin()->begin()->size() != pattern.size()) {
            pattern_vec.push_back({{pattern}});
        } else {
            auto it = pattern_vec.rbegin();
            while (it != pattern_vec.rend()) {
                auto ref = it->front();
                if (ref.size() != pattern.size()) {
                    pattern_vec.push_back({{pattern}});
                    break;
                }
                bool is_matched = true;
                for (size_t i = 0; i < pattern.size(); ++i) {
                    if (!model_comparator->match(pattern[i], ref[i])) {
                        is_matched = false;
                        break;
                    }
                }
                if (is_matched) {
                    it->push_back(pattern);
                    break;
                } else {
                    it++;
                }
            }
            if (it == pattern_vec.rend()) {
                pattern_vec.push_back({{pattern}});
            }
        }
    }
    return pattern_vec;
}

std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>>
RepeatPatternExtractor::find_repeat_patterns(const std::shared_ptr<ov::Model> &model,
                                             bool is_save_borders_only) {
    std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>> extracted_patterns;
    auto ordered_ops = model->get_ordered_ops();
    if (ordered_ops.size() < 2) {
        return extracted_patterns;
    }
    auto matched_nodes_pattern = model_comparator->get_matched_op_patterns(ordered_ops);

    for (size_t i = 0; i < matched_nodes_pattern.size(); ++i) {
        auto matched_nodes = matched_nodes_pattern[i];
        if (matched_nodes.size() < 2 || i > 0 && matched_nodes.size() == matched_nodes_pattern[i - 1].size()) {
            continue;
        }
        for (auto& nodes_vector : get_patterns_by_nodes(matched_nodes, ordered_ops)) {
            try {
                if (nodes_vector.size() < 1) {
                    continue;
                }
                auto extracted_pattern = ov::util::generate_model(nodes_vector.front(), is_save_const, is_save_borders_only);
                auto extracted_model = extracted_pattern.first;
                auto extracted_input_info = extracted_pattern.second;
                if (extracted_model == nullptr) {
                    continue;
                }
                bool is_insert_res = true;
                if (is_recursive_extraction) {
                    auto tmp_extracted_patterns = find_repeat_patterns(extracted_model, is_save_borders_only);
                    if (!tmp_extracted_patterns.empty()) {
                        is_insert_res = false;
                        update_extractor_cache(extracted_patterns, tmp_extracted_patterns);
                    }
                }
                if (is_insert_res) {
                    update_extractor_cache(extracted_patterns,
                                           extracted_model,
                                           nodes_vector,
                                           extracted_input_info);
                }
            } catch(std::exception& e) {
                if (std::string(e.what()).find("Incorrect node number to create model!") == std::string::npos) {
                    // std::cout << "[ WARNING ] Impossible to generate network and add to GraphCache: " <<e.what() << std::endl;
                }
            }
        }
        if (is_extract_body) {
            for (const auto& matched_node_idx : matched_nodes) {
                const auto& matched_node = ordered_ops[matched_node_idx];
                if (ov::as_type_ptr<ov::op::v0::TensorIterator>(matched_node)) {
                    auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(matched_node);
                    auto ti_body = ti->get_function();
                    auto secondary_patterns = find_repeat_patterns(ti_body, is_save_borders_only);
                    update_extractor_cache(extracted_patterns, secondary_patterns);
                } else if (ov::as_type_ptr<ov::op::v5::Loop>(matched_node)) {
                    auto loop = ov::as_type_ptr<ov::op::v5::Loop>(matched_node);
                    auto loop_body = loop->get_function();
                    auto secondary_patterns = find_repeat_patterns(loop_body, is_save_borders_only);
                    update_extractor_cache(extracted_patterns, secondary_patterns);
                } else if (ov::as_type_ptr<ov::op::v8::If>(matched_node)) {
                    auto if_op = ov::as_type_ptr<ov::op::v8::If>(matched_node);
                    std::vector<std::shared_ptr<ov::Model>> bodies;
                    for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                        auto if_body = if_op->get_function(i);
                        auto secondary_patterns = find_repeat_patterns(if_body, is_save_borders_only);
                        update_extractor_cache(extracted_patterns, secondary_patterns);
                    }
                } else {
                    break;
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

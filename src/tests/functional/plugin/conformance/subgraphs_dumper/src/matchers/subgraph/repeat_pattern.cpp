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
                                           model_comparator->get_matched_ops(pattern, cached_pattern));
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

void
get_nodes_to_check(std::unordered_set<std::string>& nodes_to_check,
                   const std::shared_ptr<ov::Node>& node) {
    const auto node_name = node->get_friendly_name();
    if (nodes_to_check.empty()) {
        nodes_to_check.insert(node_name);
    }
    for (size_t out_idx = 0; out_idx < node->outputs().size(); ++out_idx) {
        for (const auto& out : node->get_output_target_inputs(out_idx)) {
            const auto& output_node = out.get_node()->shared_from_this();
            if (ov::op::util::is_output(output_node)) {
                return;
            }
            if (!nodes_to_check.count(output_node->get_friendly_name())) {
                nodes_to_check.insert(output_node->get_friendly_name());
                get_nodes_to_check(nodes_to_check, output_node);
            }
        }
    }
    return;
}

std::unordered_map<std::shared_ptr<ov::Node>, std::vector<size_t>>
RepeatPatternExtractor::get_matched_nodes(const ov::NodeVector& ordered_ops) {
    std::unordered_map<std::shared_ptr<ov::Node>, std::vector<size_t>> matched_nodes;
    for (size_t node_idx = 0; node_idx < ordered_ops.size(); ++node_idx) {
        bool is_matched = false;
        for (auto& matched_node : matched_nodes) {
            if (model_comparator->match(matched_node.first, ordered_ops[node_idx])) {
                matched_node.second.push_back(node_idx);
                is_matched = true;
                break;
            }
        }
        if (!is_matched && !ov::util::is_node_to_skip(ordered_ops[node_idx])) {
            matched_nodes.insert({ordered_ops[node_idx], {node_idx}});
        }
    }
    std::vector<std::shared_ptr<ov::Node>> to_remove;
    for (auto& matched_node : matched_nodes) {
        if (matched_node.second.size() < 3) {
            to_remove.push_back(matched_node.first);
        }
    }
    for (const auto& node_to_remove : to_remove) {
        matched_nodes.erase(node_to_remove);
    }
    return matched_nodes;
}

// std::vector<std::vector<ov::NodeVector>>
std::vector<std::vector<std::vector<size_t>>>
RepeatPatternExtractor::get_patterns_by_nodes(const std::vector<size_t>& start_op_vec,
                                              const ov::NodeVector& ordered_ops) {
    if (start_op_vec.size() < 2 || ordered_ops.size() < 3) {
        return {{}};
    }
    auto pattern_cnt = start_op_vec.size();
    // std::vector<ov::NodeVector> patterns(pattern_cnt);
    std::vector<std::vector<size_t>> patterns(pattern_cnt);
    for (size_t pattern_idx = 0; pattern_idx < pattern_cnt; ++pattern_idx) {
        std::unordered_set<std::string> nodes_to_check;
        const auto& start_op_idx = start_op_vec[pattern_idx];
        get_nodes_to_check(nodes_to_check, ordered_ops[start_op_idx]);
        // const ov::NodeVector buf_ordered_ops(ordered_ops.begin() + start_op_idx, ordered_ops.end());
        std::vector<size_t> buf_ordered_ops(ordered_ops.size() - start_op_idx);
        std::iota(buf_ordered_ops.begin(), buf_ordered_ops.end(), start_op_idx);
        // for (const auto& op : buf_ordered_ops) {
        for (const auto& op_idx : buf_ordered_ops) {
            // if (nodes_to_check.count(op->get_friendly_name())) {
            if (nodes_to_check.count(ordered_ops[op_idx]->get_friendly_name())) {
                patterns[pattern_idx].push_back(op_idx);
            }
        }
    }
    {
        // std::vector<ov::NodeVector> potential_patterns(pattern_cnt);
        std::vector<std::vector<size_t>> potential_patterns(pattern_cnt);
        for (size_t i_orig = 0; i_orig < pattern_cnt - 1; ++i_orig) {
            for (size_t i_ref = i_orig + 1; i_ref < pattern_cnt; ++i_ref) {
                auto intersection_len = std::min(patterns[i_orig].size(), patterns[i_ref].size());
                // if (potential_patterns[i_orig].size() > intersection_len ||
                //     potential_patterns[i_ref].size() > intersection_len) {
                //     continue;
                // }
                std::vector<size_t> pattern_orig(intersection_len, 0);
                std::vector<size_t> pattern_ref(intersection_len, 0);
                std::unordered_set<std::shared_ptr<ov::Node>> orig_set;
                std::unordered_set<std::shared_ptr<ov::Node>> ref_set;
                for (size_t j = 0; j < intersection_len; ++j) {
                    if (model_comparator->match(ordered_ops[patterns[i_orig][j]], ordered_ops[patterns[i_ref][j]])) {
                        if (j != 0) {
                            bool is_input_matched = false;
                            for (size_t input_idx = 0; input_idx < ordered_ops[patterns[i_orig][j]]->inputs().size(); ++input_idx) {
                                auto in_orig = ordered_ops[patterns[i_orig][j]]->get_input_node_shared_ptr(input_idx);
                                auto in_ref = ordered_ops[patterns[i_ref][j]]->get_input_node_shared_ptr(input_idx);
                                // if (std::find(pattern_orig.begin(), pattern_orig.end(), in_orig) != pattern_orig.end() &&
                                //     std::find(pattern_ref.begin(), pattern_ref.end(), in_ref) != pattern_ref.end()) {
                                //     is_input_matched = true;
                                //     break;
                                // }
                                if (orig_set.count(in_orig) && ref_set.count(in_ref)) {
                                    is_input_matched = true;
                                    break;
                                }
                            }
                            if (!is_input_matched) {
                                break;
                            }
                        }
                        pattern_orig[j] = patterns[i_orig][j];
                        pattern_ref[j] = patterns[i_ref][j];
                        orig_set.insert(ordered_ops[patterns[i_orig][j]]);
                        ref_set.insert(ordered_ops[patterns[i_ref][j]]);
                    }
                }
                std::vector<size_t> orig, ref;
                for (size_t node_idx = 0; node_idx < pattern_orig.size(); ++node_idx) {
                    if (pattern_orig[node_idx] != 0) {
                        orig.emplace_back(pattern_orig[node_idx]);
                    }
                    if (pattern_ref[node_idx] != 0) {
                        ref.emplace_back(pattern_ref[node_idx]);
                    }
                }
                // if (potential_patterns[i_orig].size() < orig.size()) {
                //     potential_patterns[i_orig] = orig;
                // }
                // if (potential_patterns[i_ref].size() < ref.size()) {
                //     potential_patterns[i_ref] = ref;
                // }
                if ((potential_patterns[i_orig].size() > orig.size() || potential_patterns[i_orig].empty()) && orig.size() > 1) {
                    potential_patterns[i_orig] = orig;
                }
                if ((potential_patterns[i_ref].size() > ref.size()|| potential_patterns[i_ref].empty()) && ref.size() > 1) {
                    potential_patterns[i_ref] = ref;
                }
            }
        }
        std::sort(potential_patterns.begin(), potential_patterns.end(), [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
            return a.size() > b.size();
        });

        while (potential_patterns.rbegin()->size() < 2 && !potential_patterns.empty()) {
            potential_patterns.pop_back();
        }
        patterns = potential_patterns;
    }

    // std::vector<std::vector<ov::NodeVector>> pattern_vec;
    std::vector<std::vector<std::vector<size_t>>> pattern_vec;
    for (size_t pattern_idx = 0; pattern_idx < patterns.size(); ++pattern_idx) {
        const auto& pattern = patterns[pattern_idx];
        if (pattern_vec.empty()) {
            pattern_vec.push_back({{pattern}});
        } else if (pattern_vec.rbegin()->begin()->size() != pattern.size()) {
            pattern_vec.push_back({{pattern}});
        } else {
            auto ref = pattern_vec.rbegin()->front();
            bool is_matched = true;
            for (size_t i = 0; i < pattern.size(); ++i) {
                if (!model_comparator->match(ordered_ops[pattern[i]], ordered_ops[ref.at(i)])) {
                    pattern_vec.push_back({{pattern}});
                    is_matched = false;
                    break;
                }
            }
            if (is_matched)
                pattern_vec.rbegin()->push_back(pattern);
        }
    }
    return pattern_vec;
}

std::vector<ov::NodeVector>
a(const std::vector<std::vector<size_t>>& b, const ov::NodeVector& ordered_ops) {
    std::vector<ov::NodeVector> res(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        ov::NodeVector r(b[i].size());
        for (size_t j = 0; j < r.size(); ++j) {
            r[j] = ordered_ops[b[i][j]];
        }
        res[i] = r;
    }
    return res;
}

std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>>
RepeatPatternExtractor::find_repeat_patterns(const std::shared_ptr<ov::Model> &model,
                                             bool is_save_borders_only) {
    std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>> extracted_patterns;
    auto ordered_ops = model->get_ordered_ops();
    auto op_cnt = ordered_ops.size();
    auto matched_nodes = get_matched_nodes(ordered_ops);
    // std::cout << "MATCHED: " << matched_nodes.size() << std::endl;
    size_t v = 0;

    for (const auto& matched_node : matched_nodes) {
        // std::cout << "CNT: " << v++ << std::endl;
        if (matched_node.second.size() < 2) {
            continue;
        }
        // auto start_time = std::chrono::system_clock::now();
        auto patterns = get_patterns_by_nodes(matched_node.second, ordered_ops);
        // auto end_time = std::chrono::system_clock::now();
        // std::chrono::duration<double> duration = end_time - start_time;
        // std::cout << "Get patterns " << duration.count() << "s" << std::endl;
        // start_time = std::chrono::system_clock::now();
        for (auto& nodes_vector : patterns) {
            try {
                auto d = a(nodes_vector, ordered_ops);
                auto extracted_pattern = ov::util::generate_model(d.front(), is_save_const, is_save_borders_only);
                const auto& extracted_model = extracted_pattern.first;
                const auto& extracted_input_info = extracted_pattern.second;
                if (extracted_model == nullptr) {
                    continue;
                }
                update_extractor_cache(extracted_patterns,
                                        extracted_model,
                                        d,
                                        extracted_input_info);
            } catch(std::exception& e) {
                if (std::string(e.what()).find("Incorrect node number to create model!") == std::string::npos) {
                    // std::cout << "[ WARNING ] Impossible to generate network and add to GraphCache: " <<e.what() << std::endl;
                }
            }
        }

        // end_time = std::chrono::system_clock::now();
        // duration = end_time - start_time;
        // std::cout << "Recover model " << duration.count() << "s" << std::endl;

        // if (is_extract_body) {
        //     if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(matched_node.first)) {
        //         auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(matched_node.first);
        //         auto ti_body = ti->get_function();
        //         auto secondary_patterns = find_repeat_patterns(ti_body, is_save_borders_only);
        //         update_extractor_cache(extracted_patterns, secondary_patterns);
        //     } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(matched_node.first)) {
        //         auto loop = ov::as_type_ptr<ov::op::v5::Loop>(matched_node.first);
        //         auto loop_body = loop->get_function();
        //         auto secondary_patterns = find_repeat_patterns(loop_body, is_save_borders_only);
        //         update_extractor_cache(extracted_patterns, secondary_patterns);
        //     } else if (std::dynamic_pointer_cast<ov::op::v8::If>(matched_node.first)) {
        //         auto if_op = ov::as_type_ptr<ov::op::v8::If>(matched_node.first);
        //         std::vector<std::shared_ptr<ov::Model>> bodies;
        //         for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
        //             auto if_body = if_op->get_function(i);
        //             auto secondary_patterns = find_repeat_patterns(if_body, is_save_borders_only);
        //             update_extractor_cache(extracted_patterns, secondary_patterns);
        //         }
        //     }
        // }
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

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
                ov::util::align_input_info(pattern, cached_pattern,
                                           pattern_in_info, cached_in_info,
                                           model_comparator->get_matched_ops(pattern, cached_pattern));
                extracted_pattern.push_back({ pattern, pattern_node_vector, pattern_in_info });
                // std::cout << "UPDATE" << std::endl;
                return;
            } catch(std::exception) {}
        }
    }
    extracted_patterns.push_back({{ pattern, pattern_node_vector, pattern_in_info }});
    // std::cout << "ADD" << std::endl;
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

// void
// RepeatPatternExtractor::get_node_queque(ov::NodeVector& queue,
//                 const std::shared_ptr<ov::Node>& node) {
//     if (queue.empty()) {
//         queue.push_back(node);
//     // if (node->get_friendly_name() == "ShapeOf_1658") {
//     //     auto a = 0;
//     // }
//     }
//     for (size_t out_idx = 0; out_idx < node->outputs().size(); ++out_idx) {
//         for (const auto& out : node->get_output_target_inputs(out_idx)) {
//             const auto& output_node = out.get_node()->shared_from_this();
//             if (ov::op::util::is_output(output_node)) {
//                 return;
//             }
//             {
//                 auto it_insert = std::find(queue.begin(), queue.end(), output_node);
//                 while (it_insert != queue.end()) {
//                     queue.erase(it_insert);
//                     it_insert = std::find(queue.begin(), queue.end(), output_node);
//                 }
//             }
//             queue.push_back(output_node);
//             get_node_queque(queue, output_node);
//         }
//     }
//     return;
// }

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

std::vector<ov::NodeVector>
RepeatPatternExtractor::get_patterns_by_nodes(const std::vector<size_t>& start_op_vec,
                                              const ov::NodeVector& ordered_ops) {
    auto pattern_cnt = start_op_vec.size();
    std::vector<std::vector<RepeatPatternExtractor::NodePair>> patterns(pattern_cnt);
    size_t max_node_cnt = 0;
    for (size_t pattern_idx = 0; pattern_idx < pattern_cnt; ++pattern_idx) {
        std::unordered_set<std::string> nodes_to_check;
        const auto& start_op_idx = start_op_vec[pattern_idx];
        get_nodes_to_check(nodes_to_check, ordered_ops[start_op_idx]);
        const ov::NodeVector buf_ordered_ops(ordered_ops.begin() + start_op_idx, ordered_ops.end());
        for (const auto& op : buf_ordered_ops) {
            if (nodes_to_check.count(op->get_friendly_name())) {
                auto& pattern = patterns[pattern_idx];
                std::vector<size_t> input_cnt;
                for (size_t i = 0; i < op->inputs().size(); ++i) {
                    const auto& input_node = op->get_input_node_shared_ptr(i);
                    auto it = std::find_if(pattern.begin(), pattern.end(),
                        [&input_node](const RepeatPatternExtractor::NodePair& item) {
                            return input_node == item.first;
                        });
                    if (it != pattern.end() && !ov::util::is_node_to_skip(op)) {
                        input_cnt.push_back(std::distance(pattern.begin(), it));
                    }
                }
                pattern.push_back({ op, input_cnt });
                const auto pattern_node_cnt = patterns[pattern_idx].size();
                if (max_node_cnt < pattern_node_cnt) {
                    max_node_cnt = pattern_node_cnt;
                }
            }
        }
    }

    // std::cout << "SIZE: " << start_op_vec.size()  << " " << max_node_cnt << std::endl;

    std::vector<std::vector<bool>> mask(pattern_cnt, std::vector<bool>(max_node_cnt, false));
    auto is_valid_input = [&patterns, &mask](size_t i, size_t j) {
        if (patterns[i][j].second.size() == 0 && j == 0) {
            return true;
        }
        for (const auto& in_idx : patterns[i][j].second) {
            if (mask[i][in_idx]) {
                return true;
            }
        }
        return false;
    };


    for (size_t j = 0; j < max_node_cnt; ++j) {
        for (size_t i_orig = 0; i_orig < pattern_cnt - 1; ++i_orig) {
            if (j >= patterns[i_orig].size())
                continue;
            if (!is_valid_input(i_orig, j))
                continue;
            for (size_t i_ref = i_orig + 1; i_ref < pattern_cnt; ++i_ref) {
                if (mask[i_orig][j] && mask[i_ref][j])
                    continue;
                if (j >= patterns[i_ref].size())
                    continue;
                if (!is_valid_input(i_ref, j))
                    continue;
                if (patterns[i_orig][j].first == patterns[i_ref][j].first) {
                    continue;
                }
                if (model_comparator->match(patterns[i_orig][j].first, patterns[i_ref][j].first)) {
                    mask[i_orig][j] = true;
                    mask[i_ref][j] = true;
                }
            }
        }
    }

    std::vector<std::unordered_set<std::shared_ptr<ov::Node>>> result_patterns(pattern_cnt);
    for (size_t i_orig = 0; i_orig < pattern_cnt - 1; ++i_orig) {
        for (size_t i_ref = i_orig + 1; i_ref < pattern_cnt; ++i_ref) {
            std::unordered_set<std::shared_ptr<ov::Node>> orig_nodes, ref_nodes;
            const auto intersection_len = std::min(patterns[i_orig].size(), patterns[i_ref].size());
            for (size_t j = 0; j < intersection_len; ++j) {
                if (!mask[i_orig][j] || !mask[i_ref][j]) {
                    continue;
                }
                if (model_comparator->match(patterns[i_orig][j].first, patterns[i_ref][j].first)) {
                    bool has_input = orig_nodes.empty();
                    if (!has_input) {
                        for (size_t input_idx = 0; input_idx < patterns[i_orig][j].first->inputs().size(); ++input_idx) {
                            const auto& input_node_orig = patterns[i_orig][j].first->get_input_node_shared_ptr(input_idx);
                            if (orig_nodes.count(input_node_orig)) {
                                has_input = true;
                                break;
                            }
                        }
                    }
                    if (has_input) {
                        orig_nodes.insert(patterns[i_orig][j].first);
                        ref_nodes.insert(patterns[i_ref][j].first);
                    }
                }
            }
            if (result_patterns[i_ref].size() < ref_nodes.size()) {
                result_patterns[i_ref] = ref_nodes;
            }
            if (result_patterns[i_orig].size() < orig_nodes.size()) {
                result_patterns[i_orig] = orig_nodes;
            }
        }
    }
    std::vector<ov::NodeVector> r_;
    for (const auto& r : result_patterns) {
        r_.push_back({r.begin(), r.end()});
    }
    return r_;
}

std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>>
RepeatPatternExtractor::find_repeat_patterns(const std::shared_ptr<ov::Model> &model,
                                             bool is_save_borders_only) {
    std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>> extracted_patterns;
    std::unordered_set<std::shared_ptr<ov::Node>> checked_op_pattern;
    auto ordered_ops = model->get_ordered_ops();
    auto op_cnt = ordered_ops.size();

    for (size_t idx = 0; idx < op_cnt; ++idx) {
        auto op = ordered_ops[idx];
        auto node_name = op->get_friendly_name();
        if (ov::util::is_node_to_skip(op)) {
            continue;
        }

        bool is_checked_op_pattern = false;
        for (const auto& checked_op : checked_op_pattern) {
            if (model_comparator->match(op, checked_op)) {
                is_checked_op_pattern = true;
                break;
            }
        }
        if (is_checked_op_pattern) {
            continue;
        } else {
            checked_op_pattern.insert(op);
            // std::cout << checked_op_pattern.size() << std::endl;
        }

        std::vector<size_t> pattern_start_nodes{idx};
        for (size_t i = idx+1; i < op_cnt; ++i) {
            if (model_comparator->match(op, ordered_ops[i])) {
                pattern_start_nodes.push_back(i);
            }
        }
        if (pattern_start_nodes.size() < 2) {
            continue;
        }

        auto patterns = get_patterns_by_nodes(pattern_start_nodes, ordered_ops);
        for (auto& nodes_vector : patterns) {
            try {
                auto extracted_pattern = ov::util::generate_model(nodes_vector, is_save_const, is_save_borders_only);
                const auto& extracted_model = extracted_pattern.first;
                const auto& extracted_input_info = extracted_pattern.second;
                if (extracted_model == nullptr) {
                    continue;
                }
                // if (is_recursive_extraction) {
                //     auto secondary_patterns = find_repeat_patterns(extracted_model, is_save_borders_only);
                //     if (!secondary_patterns.empty()) {
                //         update_extractor_cache(extracted_patterns, secondary_patterns);
                //     } else {
                        update_extractor_cache(extracted_patterns,
                                               extracted_model,
                                               nodes_vector,
                                               extracted_input_info);
                    // }
                // } else {
                //     update_extractor_cache(extracted_patterns,
                //                            extracted_model,
                //                            nodes_vector,
                //                            extracted_input_info);
                // }
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

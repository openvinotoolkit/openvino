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

std::vector<std::vector<RepeatPatternExtractor::NodePair>>
RepeatPatternExtractor::get_ordered_nodes(const std::vector<size_t>& start_node_vec, const ov::NodeVector& ordered_nodes) {
    auto pattern_cnt = start_node_vec.size();
    std::vector<std::vector<RepeatPatternExtractor::NodePair>> patterns(pattern_cnt);

    for (size_t pattern_idx = 0; pattern_idx < pattern_cnt; ++pattern_idx) {
        ov::NodeVector queue;
        std::unordered_set<std::string> nodes_to_check;
        get_nodes_to_check(nodes_to_check, ordered_nodes[start_node_vec[pattern_idx]]);
        for (const auto& op : ordered_nodes) {
            if (nodes_to_check.count(op->get_friendly_name())) {
                queue.push_back(op);
            }
        }
        for (const auto& node : queue) {
            std::vector<size_t> input_cnt;
            for (size_t i = 0; i < node->inputs().size(); ++i) {
                const auto& input_node = node->get_input_node_shared_ptr(i);
                auto it = std::find(queue.begin(), queue.end(), input_node);
                if (it != queue.end() && !ov::util::is_node_to_skip(node)) {
                    input_cnt.push_back(std::distance(queue.begin(), it));
                }
            }
            patterns[pattern_idx].push_back({ node, input_cnt });
        }
    }
    return patterns;
}

std::vector<ov::NodeVector>
RepeatPatternExtractor::post_process_patterns(const std::vector<std::vector<NodePair>>& patterns) {
    size_t max_node_cnt = 0;
    auto pattern_cnt = patterns.size();
    std::vector<std::vector<bool>> mask;
    {
        std::vector<size_t> pattern_node_cnt;
        for (const auto& pattern : patterns) {
            const auto pattern_size = pattern.size();
            pattern_node_cnt.push_back(pattern_size);
            if (pattern_size > max_node_cnt) {
                max_node_cnt = pattern_size;
            }
        }

        mask.resize(pattern_cnt);
        for (auto& mask_row : mask) {
            mask_row = std::vector<bool>(max_node_cnt, false);
        }

        auto is_valid_input = [&patterns, &mask](size_t i, size_t j) {
            bool is_any_input = patterns[i][j].second.size() == 0;
            if (is_any_input)
                return true;
            for (const auto& in_idx : patterns[i][j].second) {
                if (mask[i][in_idx]) {
                    is_any_input = true;
                    break;
                }
            }
            return is_any_input;
        };

        for (size_t j = 0; j < max_node_cnt; ++j) {
            for (size_t i_orig = 0; i_orig < pattern_cnt - 1; ++i_orig) {
                if (j >= pattern_node_cnt[i_orig])
                    continue;
                if (!is_valid_input(i_orig, j))
                    continue;
                for (size_t i_ref = i_orig + 1; i_ref < pattern_cnt; ++i_ref) {
                    if (mask[i_orig][j] && mask[i_ref][j])
                        continue;
                    if (j >= pattern_node_cnt[i_ref])
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
    }

    std::vector<ov::NodeVector> tmp_result_pattern;
    {
        std::map<size_t, std::vector<ov::NodeVector>> b;
        for (size_t i = 0; i < pattern_cnt; ++i) {
            ov::NodeVector tmp_buf;
            for (size_t j = 0; j < max_node_cnt; ++j) {
                if (mask[i][j]) {
                    tmp_buf.push_back(patterns[i][j].first);
                }
            }
            if (tmp_buf.size() > 1) {
                if (b.count(tmp_buf.size())) {
                    b[tmp_buf.size()].push_back(tmp_buf);
                } else {
                    b.insert({tmp_buf.size(), {tmp_buf}});
                }
            }
        }
        for (const auto a : b) {
            if (a.second.size() > 1) {
                tmp_result_pattern.insert(tmp_result_pattern.begin(), a.second.begin(), a.second.end());
            }
        }
    }

    auto pattern_size = tmp_result_pattern.size();
    std::vector<ov::NodeVector> result_pattern(pattern_size);
    for (size_t i = 0; i < pattern_size; ++i) {
        const auto& pattern = tmp_result_pattern[i];

        ov::NodeVector max_pattern, max_pattern_ref;
        auto max_pattern_size = 0;
        for (size_t j = 0; j < pattern_size; ++j) {
            if (i == j) {
                continue;
            }

            const auto& pattern_ref = tmp_result_pattern[j];
            auto max_nodes = std::min(pattern.size(), pattern_ref.size());
            if (max_nodes < max_pattern_size) {
                if (result_pattern[j].size() < max_pattern_size) {
                    result_pattern[j] = max_pattern_ref;
                }
                break;
            }
            for (size_t k = 0; k < max_nodes; ++k) {
                if (!model_comparator->match(pattern[k], pattern_ref[k]) && k > max_pattern_size) {
                    max_pattern.clear();
                    max_pattern_ref.clear();

                    auto it_end = pattern.begin();
                    std::advance(it_end, k);
                    max_pattern.insert(max_pattern.end(), pattern.begin(), it_end);

                    it_end = pattern_ref.begin();
                    std::advance(it_end, k);
                    max_pattern_ref.insert(max_pattern_ref.end(), pattern_ref.begin(), it_end);
                    max_pattern_size = k;
                }
            }
        }
        if (result_pattern[i].size() < max_pattern_size) {
            result_pattern[i] = max_pattern;
        }
    }
    std::vector<ov::NodeVector> r;
    for (size_t i = 0; i < pattern_size; ++i) {
        auto min_subgraph = result_pattern[i];
        size_t min_subgraph_len = result_pattern[i].size();

        for (size_t j = i + 1; j < pattern_size; ++j) {
            auto it = result_pattern[j].begin();
            while (it != result_pattern[j].end() &&
                   std::find(result_pattern[i].begin(), result_pattern[i].end(), *it) != result_pattern[i].end()) {
                ++it;
            }
            if (it == result_pattern[j].end()) {
                if (min_subgraph_len > result_pattern[j].size()) {
                    min_subgraph = result_pattern[j];
                    min_subgraph_len = result_pattern[j].size();
                }
            }
        }
        r.push_back(min_subgraph);
    }
    return r;
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
        auto patterns = get_ordered_nodes(pattern_start_nodes, ordered_ops);
        auto potential_patterns = post_process_patterns(patterns);
        std::list<std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>> temp_extracted_patterns;
        for (auto& nodes_vector : potential_patterns) {
            try {
                auto extracted_pattern = ov::util::generate_model(nodes_vector, is_save_const, is_save_borders_only);
                auto extracted_model = extracted_pattern.first;
                if (extracted_model == nullptr) {
                    continue;
                }
                // bool is_matched_model = false;
                // for (auto& a : temp_extracted_patterns) {
                //     auto b = model_comparator->match(extracted_pattern.first, std::get<0>(a.front()),
                //                                      extracted_pattern.second, std::get<2>(a.front()));
                //     if (b.first) {
                //         a.push_back({extracted_pattern.first, nodes_vector, extracted_pattern.second});
                //         is_matched_model = true;
                //         break;
                //     }
                // }
                // if (!is_matched_model) {
                //     temp_extracted_patterns.push_back({{extracted_pattern.first, nodes_vector, extracted_pattern.second}});
                // }

                // if (is_recursive_extraction && nodes_vector.size() > 10) {
                //     auto secondary_patterns = find_repeat_patterns(extracted_model, is_save_borders_only);
                //     if (!secondary_patterns.empty()) {
                //         // tmp_checked_op_pattern.clear();
                //         update_extractor_cache(extracted_patterns, secondary_patterns);
                //     } else {
                //         update_extractor_cache(extracted_patterns,
                //                                extracted_model,
                //                                nodes_vector,
                //                                extracted_pattern.second);
                //     }
                // } else {
                    update_extractor_cache(extracted_patterns,
                                           extracted_model,
                                           nodes_vector,
                                           extracted_pattern.second);
                // }
                // checked_op_pattern.insert(tmp_checked_op_pattern.begin(), tmp_checked_op_pattern.end());
            } catch(std::exception& e) {
                if (std::string(e.what()).find("Incorrect node number to create model!") == std::string::npos) {
                    // std::cout << "[ WARNING ] Impossible to generate network and add to GraphCache: " <<e.what() << std::endl;
                }
            }
        }
        // auto it = temp_extracted_patterns.begin();
        // while (it != temp_extracted_patterns.end()) {
        //     auto it_1 = std::next(it);
        //     while (it_1 != temp_extracted_patterns.end()) {
        //         try {
        //             auto a = model_comparator->is_subgraph(std::get<0>(it->front()), std::get<0>(it_1->front()),
        //                                                    std::get<2>(it->front()), std::get<2>(it_1->front()));
        //             if (std::get<0>(a)) {
        //                 auto b = 0;
        //             }
        //         } catch (...) {
        //             auto c = 0;
        //         }
        //         ++it_1;
        //     }
        //     ++it;
        // }

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

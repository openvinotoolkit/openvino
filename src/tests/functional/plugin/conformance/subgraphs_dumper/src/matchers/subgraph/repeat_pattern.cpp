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

std::vector<std::vector<RepeatPatternExtractor::NodePair>>
RepeatPatternExtractor::get_ordered_nodes(const ov::NodeVector& start_node_vec) {
    auto pattern_cnt = start_node_vec.size();
    std::vector<std::vector<NodePair>> patterns(pattern_cnt);

    for (size_t pattern_idx = 0; pattern_idx < pattern_cnt; ++pattern_idx) {
        if (!patterns[pattern_idx].empty()) {
            continue;
        }

        std::list<std::shared_ptr<ov::Node>> queue = { start_node_vec[pattern_idx] };
        while (!queue.empty()) {
            const auto& node = queue.front();
            for (size_t node_vec_idx = pattern_idx; node_vec_idx < pattern_cnt; ++node_vec_idx) {
                auto& node_vector = patterns[node_vec_idx];
                if (node_vector.empty()) {
                    if (node == start_node_vec[node_vec_idx]) {
                        node_vector.push_back({node, {}});
                    }
                    break;
                }

                std::vector<size_t> input_indexes;
                for (size_t in_idx = 0; in_idx < node->inputs().size(); in_idx++) {
                    const auto& input_node = node->get_input_node_shared_ptr(in_idx);
                    size_t input_idx = node_vector.size();
                    while (input_idx > 0) {
                        --input_idx;
                        if (node_vector[input_idx].first == input_node) {
                            input_indexes.push_back(input_idx);
                            break;
                        }
                    }
                }
                if (!input_indexes.empty()) {
                    node_vector.push_back({node, input_indexes});
                }
            }

            for (size_t out_idx = 0; out_idx < node->get_output_size(); ++out_idx) {
                for (const auto& target_input : node->get_output_target_inputs(out_idx)) {
                    auto output_node = target_input.get_node()->shared_from_this();
                    if (!ov::op::util::is_output(output_node)) {
                        if (std::find(queue.begin(), queue.end(), output_node) != queue.end()) {
                            continue;
                        }
                        auto it_to_insert = queue.begin();
                        for (size_t i = 0; i < output_node->inputs().size(); ++i) {
                            const auto& input_node = output_node->get_input_node_shared_ptr(i);
                            auto new_it = std::find(queue.begin(), queue.end(), input_node);
                            if (new_it != queue.end()) {
                                it_to_insert = std::max_element(new_it, it_to_insert);
                            }
                        }
                        std::advance(it_to_insert, out_idx + 1);
                        queue.insert(it_to_insert, output_node);
                    }
                }
            }
            while (queue.front() == node && !queue.empty()) {
                queue.pop_front();
            }
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
                    if (model_comparator->match(patterns[i_orig][j].first, patterns[i_ref][j].first)) {
                        mask[i_orig][j] = true;
                        mask[i_ref][j] = true;
                    }
                }
            }
        }
    }

    std::vector<ov::NodeVector> result_pattern;
    for (size_t i = 0; i < pattern_cnt; ++i) {
        ov::NodeVector tmp_buf;
        for (size_t j = 0; j < max_node_cnt; ++j) {
            if (mask[i][j]) {
                tmp_buf.push_back(patterns[i][j].first);
            }
        }
        if (tmp_buf.size() > 1) {
            result_pattern.push_back(tmp_buf);
        }
    }
    return result_pattern;
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

        ov::NodeVector pattern_start_nodes;
        for (size_t i = idx + 1; i < op_cnt; ++i) {
            if (model_comparator->match(op, ordered_ops[i])) {
                pattern_start_nodes.push_back(ordered_ops[i]);
            }
        }
        if (pattern_start_nodes.size() < 2) {
            checked_ops.insert(op_name);
            continue;
        }
        auto patterns = get_ordered_nodes(pattern_start_nodes);
        auto potential_patterns = post_process_patterns(patterns);
        for (auto& nodes_vector : potential_patterns) {
            try {
                std::unordered_set<std::string> tmp_checked_ops;
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

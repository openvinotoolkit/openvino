// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/model_comparator.hpp"
#include "utils/model.hpp"

using namespace ov::util;

std::shared_ptr<ModelComparator> ModelComparator::m_instance = nullptr;

void ModelComparator::set_match_coefficient(float _match_coefficient) {
    if (_match_coefficient  < 0 || _match_coefficient > 1) {
        throw std::runtime_error("[ ERROR ] Match coefficient should be from 0 to 1!");
    }
    match_coefficient = _match_coefficient;
}

void ModelComparator::set_shape_strict_match(bool in_is_shape_strict_match) {
    m_manager.set_shape_strict_match(in_is_shape_strict_match);
}

void ModelComparator::set_match_attributes(bool match_attributes) {
    m_manager.set_match_attributes(match_attributes);
}

void ModelComparator::set_match_in_types(bool match_in_types) {
    m_manager.set_match_in_types(true);
}

inline ModelComparator::IsSubgraphTuple
prepare_is_subgraph_result(bool is_subgraph,
                           const std::shared_ptr<ov::Model>& subgraph,
                           const std::shared_ptr<ov::Model>& graph,
                           const std::unordered_map<std::string, std::string>& matched_ops) {
    return std::make_tuple(is_subgraph, subgraph, graph, matched_ops);
}

ModelComparator::IsSubgraphTuple
ModelComparator::is_subgraph(const std::shared_ptr<ov::Model> &model,
                             const std::shared_ptr<ov::Model> &ref_model) const {
    auto in_info = ov::util::get_input_info_by_model(model);
    auto in_info_ref = ov::util::get_input_info_by_model(ref_model);
    size_t ordered_ops_cnt = model->get_ordered_ops().size() - in_info.size() - model->get_results().size(),
           ref_ordered_ops_cnt = ref_model->get_ordered_ops().size() - in_info_ref.size() - ref_model->get_results().size();
    bool is_model = ordered_ops_cnt > ref_ordered_ops_cnt;
    size_t subgraph_to_check_ops_cnt;
    std::shared_ptr<ov::Model> graph = nullptr, subgraph = nullptr;
    if (is_model) {
        graph = model;
        subgraph = ref_model;
        subgraph_to_check_ops_cnt = ref_ordered_ops_cnt;
    } else {
        graph = ref_model;
        subgraph = model;
        subgraph_to_check_ops_cnt = ordered_ops_cnt;
    }
    auto matched_op_names = get_matched_ops_in_graphs(subgraph, graph);
    return prepare_is_subgraph_result(matched_op_names.size() == subgraph_to_check_ops_cnt, subgraph, graph, matched_op_names);
}

bool
ModelComparator::match(const std::shared_ptr<ov::Node> &node,
                       const std::shared_ptr<ov::Node> &ref_node) const {
    return m_manager.match(node, ref_node);
}

bool
ModelComparator::match(const std::shared_ptr<ov::Model> &model,
                       const std::shared_ptr<ov::Model> &ref_model) const {
    std::vector<std::shared_ptr<ov::Node>> ordered_ops = model->get_ordered_ops(),
                                           ref_ordered_ops = ref_model->get_ordered_ops();
    if (ordered_ops.size() != ref_ordered_ops.size()) {
        return false;
    }
    size_t matched_op_cnt = 0, total_op_cnt = ordered_ops.size();
    size_t matched_op_cnt_required = round(match_coefficient * total_op_cnt);
    for (size_t i = 0; i < total_op_cnt; ++i) {
        if (m_manager.match(ordered_ops[i], ref_ordered_ops[i])) {
            ++matched_op_cnt;
        }
        if (matched_op_cnt >= matched_op_cnt_required) {
            return true;
        }
    }
    return false;
}

ModelComparator::ExtractedSubgraphTuple
ModelComparator::is_subgraph(const std::shared_ptr<ov::Model> &model,
                             const std::shared_ptr<ov::Model> &ref_model,
                             const std::map<std::string, InputInfo> &in_info,
                             const std::map<std::string, InputInfo> &in_info_ref) {
    ModelComparator::ExtractedSubgraphTuple res = { false, nullptr, nullptr, {}, {} };
    m_manager.set_match_in_types(true);
    auto extractor_res = is_subgraph(model, ref_model);
    if (std::get<0>(extractor_res)) {
        std::map<std::string, InputInfo> graph_in_info, subgraph_in_info;
        std::shared_ptr<ov::Model> subgraph = nullptr, graph = nullptr;
        if (std::get<1>(extractor_res) == model && std::get<2>(extractor_res) == ref_model) {
            subgraph = model;
            subgraph_in_info = in_info;
            graph = ref_model;
            graph_in_info = in_info_ref;
        } else if (std::get<1>(extractor_res) == ref_model && std::get<2>(extractor_res) == model) {
            subgraph = ref_model;
            subgraph_in_info = in_info_ref;
            graph = model;
            graph_in_info = in_info;
        } else {
            throw std::runtime_error("Generated models are incompatible with original ones!");
        }
        try {
            auto subgraph_in_info_new = ov::util::align_input_info(subgraph, graph,
                                                                    subgraph_in_info, graph_in_info,
                                                                    get_matched_ops_in_graphs(subgraph, graph));
            res = { true, subgraph, graph, subgraph_in_info_new, graph_in_info };
        } catch(std::exception) {}
    }
    m_manager.set_match_in_types(false);
    return res;
}

std::unordered_map<std::string, std::string>
ModelComparator::get_matched_ops_in_graphs(const std::shared_ptr<ov::Model>& subgraph,
                                           const std::shared_ptr<ov::Model>& graph,
                                           bool is_check_inputs) const {
    std::unordered_map<std::string, std::string> matched_op_names;
    std::unordered_set<std::string> checked_op;
    const auto subgraph_to_check_ops = subgraph->get_ordered_ops();
    const auto graph_to_check_ops = graph->get_ordered_ops();
    for (const auto& subgraph_op : subgraph_to_check_ops) {
        for (const auto& graph_op : graph_to_check_ops) {
            if (ov::util::is_node_to_skip(subgraph_op) ||
                ov::util::is_node_to_skip(graph_op)) {
                continue;
            }
            if (match(subgraph_op, graph_op) && !checked_op.count(graph_op->get_friendly_name())) {
                matched_op_names.insert({subgraph_op->get_friendly_name(), graph_op->get_friendly_name()});
                checked_op.insert(graph_op->get_friendly_name());
                if (is_check_inputs) {
                    for (size_t idx = 0; idx < graph_op->inputs().size(); ++idx) {
                        auto graph_in = graph_op->get_input_node_shared_ptr(idx);
                        auto subgraph_in = subgraph_op->get_input_node_shared_ptr(idx);
                        if (ov::util::is_node_to_skip(graph_in) && ov::util::is_node_to_skip(subgraph_in)) {
                            if (match(subgraph_in, graph_in)) {
                                matched_op_names.insert({subgraph_in->get_friendly_name(), graph_in->get_friendly_name()});
                                checked_op.insert(graph_in->get_friendly_name());
                            }
                        }
                    }
                }
                break;
            }
        }
    }
    return matched_op_names;
}

std::pair<bool, std::map<std::string, ov::conformance::InputInfo>>
ModelComparator::match(const std::shared_ptr<ov::Model> &model,
                       const std::shared_ptr<ov::Model> &model_ref,
                       const std::map<std::string, InputInfo> &in_info,
                       const std::map<std::string, InputInfo> &in_info_ref) {
    try {
        if (match(model, model_ref)) {
            auto new_input_info = ov::util::align_input_info(model, model_ref,
                                                             in_info, in_info_ref,
                                                             get_matched_ops_in_graphs(model, model_ref, true));
            return {true, new_input_info};
        }
    } catch (std::exception) {}
    return {false, {}};
}

std::vector<std::vector<size_t>>
ModelComparator::get_matched_op_patterns(const ov::NodeVector& ordered_ops) {
    std::vector<std::vector<size_t>> matched_nodes;
    for (size_t node_idx = 0; node_idx < ordered_ops.size(); ++node_idx) {
        bool is_matched = false;
        for (auto& matched_node_idx : matched_nodes) {
            if (match(ordered_ops[matched_node_idx.front()], ordered_ops[node_idx])) {
                matched_node_idx.push_back(node_idx);
                is_matched = true;
                break;
            }
        }
        if (!is_matched && !ov::util::is_node_to_skip(ordered_ops[node_idx])) {
            matched_nodes.push_back({node_idx});
        }
    }
    std::sort(matched_nodes.begin(), matched_nodes.end(),
             [](const std::vector<size_t>& a, const std::vector<size_t>& b){ return a.size() > b.size(); });
    while (!matched_nodes.empty() && matched_nodes.rbegin()->size() == 1) {
        matched_nodes.pop_back();
    }
    return matched_nodes;
}
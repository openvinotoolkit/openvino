// Copyright (C) 2018-2023 Intel Corporation
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

inline ModelComparator::IsSubgraphTuple
prepare_is_subgraph_result(bool is_subgraph,
                           const std::shared_ptr<ov::Model>& subgraph,
                           const std::shared_ptr<ov::Model>& graph,
                           const std::unordered_map<std::string, std::string>& matched_ops) {
    return is_subgraph ?
           std::make_tuple(is_subgraph, subgraph, graph, matched_ops) :
           std::make_tuple(is_subgraph, nullptr, nullptr, std::unordered_map<std::string, std::string>());
}

ModelComparator::IsSubgraphTuple
ModelComparator::is_subgraph(const std::shared_ptr<ov::Model> &model,
                             const std::shared_ptr<ov::Model> &ref_model) const {
    std::vector<std::shared_ptr<ov::Node>> ordered_ops = model->get_ordered_ops(),
                                           ref_ordered_ops = ref_model->get_ordered_ops();
    bool is_model = ordered_ops.size() > ref_ordered_ops.size();
    ov::NodeVector graph_to_check_ops, subgraph_to_check_ops;
    std::shared_ptr<ov::Model> graph = nullptr, subgraph = nullptr;
    if (is_model) {
        graph_to_check_ops = ordered_ops;
        subgraph_to_check_ops = ref_ordered_ops;
        graph = model;
        subgraph = ref_model;
    } else {
        graph_to_check_ops = ref_ordered_ops;
        subgraph_to_check_ops = ordered_ops;
        graph = ref_model;
        subgraph = model;
    }
    auto matched_op_names = get_matched_ops(subgraph, graph);
    return prepare_is_subgraph_result(matched_op_names.size() == subgraph_to_check_ops.size(), subgraph, graph, matched_op_names);
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
    auto extractor_res = is_subgraph(model, ref_model);
    if (std::get<0>(extractor_res)) {
        std::map<std::string, InputInfo> graph_in_info, subgraph_in_info;
        std::shared_ptr<ov::Model> subgraph = nullptr, graph = nullptr;
        // if (model == subgraph && ref_model == graph)
        if (std::get<1>(extractor_res) == model && std::get<2>(extractor_res) == ref_model) {
            subgraph = model;
            subgraph_in_info = in_info;
            graph = ref_model;
            graph_in_info = in_info_ref;
        // else if (subgraph == ref_model && graph = model)
        } else if (std::get<1>(extractor_res) == ref_model && std::get<2>(extractor_res) == model) {
            subgraph = ref_model;
            subgraph_in_info = in_info_ref;
            graph = model;
            graph_in_info = in_info;
        } else {
            throw std::runtime_error("Generated models are incompatible with original ones!");
        }
        try {
            subgraph_in_info = ov::util::align_input_info(subgraph, graph,
                                                          subgraph_in_info, graph_in_info,
                                                          get_matched_ops(subgraph, graph));
            return { true, subgraph, graph, subgraph_in_info, graph_in_info };
        } catch(std::exception) {}
    }
    return { false, nullptr, nullptr, {}, {} };
}

std::unordered_map<std::string, std::string>
ModelComparator::get_matched_ops(const std::shared_ptr<ov::Model>& subgraph,
                                 const std::shared_ptr<ov::Model>& graph) const {
    std::unordered_map<std::string, std::string> matched_op_names;
    std::set<std::string> checked_op;
    const auto subgraph_to_check_ops = subgraph->get_ordered_ops();
    const auto graph_to_check_ops = graph->get_ordered_ops();
    for (const auto& subgraph_op : subgraph_to_check_ops) {
        for (const auto& graph_op : graph_to_check_ops) {
            if (m_manager.match(subgraph_op, graph_op) && !checked_op.count(graph_op->get_friendly_name())) {
                matched_op_names.insert({subgraph_op->get_friendly_name(), graph_op->get_friendly_name()});
                checked_op.insert(graph_op->get_friendly_name());
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
                                                             get_matched_ops(model, model_ref));
            return {true, new_input_info};
        }
    } catch (std::exception) {}
    return {false, {}};
}
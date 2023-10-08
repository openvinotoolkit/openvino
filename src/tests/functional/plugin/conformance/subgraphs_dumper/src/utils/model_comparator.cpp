// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/model_comparator.hpp"
#include "utils/node.hpp"

using namespace ov::tools::subgraph_dumper;

std::shared_ptr<ModelComparator> ModelComparator::m_instance = nullptr;

std::map<std::string, InputInfo>
ModelComparator::align_input_info(const std::shared_ptr<ov::Model>& model,
                                  const std::shared_ptr<ov::Model>& model_ref,
                                  const std::map<std::string, InputInfo>& in_info,
                                  const std::map<std::string, InputInfo>& in_info_ref,
                                  const std::map<std::string, std::string> &matched_op) {
    std::map<std::string, InputInfo> new_input_info = in_info;
    bool is_update_required = false;
    for (const auto& in_info_item : in_info_ref) {
        if (!in_info.count(in_info_item.first)) {
            is_update_required = true;
            break;
        } else if (in_info.at(in_info_item.first).is_const != in_info_item.second.is_const) {
            throw std::runtime_error("Impossible to update input info!!!");
        }
    }
    if (is_update_required) {
        // align matched model names
        auto ref_model_ops = model_ref->get_ordered_ops();
        auto model_ops = model->get_ordered_ops();
        size_t ref_ordered_ops_size = ref_model_ops.size();
        size_t ordered_ops_size = model_ops.size();
        if (ref_ordered_ops_size != ordered_ops_size && matched_op.empty()) {
            throw std::runtime_error("Matched models can not be compared according different op numbers!");
        }
        for (size_t i = 0; i < ref_ordered_ops_size; ++i) {
            auto model_op_name = i < ordered_ops_size ? model_ops[i]->get_friendly_name() : "";
            auto model_ref_op_name = ref_model_ops[i]->get_friendly_name();
            if (!in_info_ref.count(model_ref_op_name) && !in_info.count(model_op_name)) {
                continue;
            }
            auto input_info = matched_op.empty() ? new_input_info[model_op_name] : in_info_ref.at(model_ref_op_name);
            std::string input_name = matched_op.count(model_ref_op_name) ? matched_op.at(model_ref_op_name) : model_op_name;
            if (new_input_info.count(input_name)) {
                if (input_info.is_const != in_info_ref.at(model_ref_op_name).is_const) {
                    throw std::runtime_error("Impossible to update input info!!!");
                }
                if (!matched_op.empty()) {
                    input_info = new_input_info.at(input_name);
                }
                new_input_info.erase(input_name);
            }
            new_input_info.insert({ model_ref_op_name, input_info });
        }
    }
    return new_input_info;
}


inline ModelComparator::IsSubgraphTuple
prepare_is_subgraph_result(bool is_subgraph,
                           const std::shared_ptr<ov::Model>& graph,
                           const std::shared_ptr<ov::Model>& subgraph,
                           const std::map<std::string, std::string>& matched_ops) {
    return is_subgraph ?
           std::make_tuple(is_subgraph, graph, subgraph, matched_ops) :
           std::make_tuple(is_subgraph, nullptr, nullptr, std::map<std::string, std::string>());
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
    std::map<std::string, std::string> matched_op_names;

    auto graph_it = graph_to_check_ops.begin(), subgraph_it = subgraph_to_check_ops.begin();
    while (graph_it != graph_to_check_ops.end() && subgraph_it != subgraph_to_check_ops.end()) {
        if (m_manager.match(*graph_it, *subgraph_it)) {
            matched_op_names.insert({ (*graph_it)->get_friendly_name(), (*subgraph_it)->get_friendly_name()});
            ++subgraph_it;
        }
        ++graph_it;
    }
    return prepare_is_subgraph_result(subgraph_it == subgraph_to_check_ops.end(), graph, subgraph, matched_op_names);
}

bool
ModelComparator::match(const std::shared_ptr<ov::Node> &node,
                       const std::shared_ptr<ov::Node> &ref_node) {
    return m_manager.match(node, ref_node);
}

bool
ModelComparator::match(const std::shared_ptr<ov::Model> &model,
                       const std::shared_ptr<ov::Model> &ref_model) const {
    bool res = m_comparator.compare(model, ref_model).valid;
    if (res) {
        return res;
    }
    std::vector<std::shared_ptr<ov::Node>> ordered_ops = model->get_ordered_ops(),
                                           ref_ordered_ops = ref_model->get_ordered_ops();
    if (ordered_ops.size() != ref_ordered_ops.size()) {
        return false;
    }
    size_t matched_op_cnt = 0, total_op_cnt = ordered_ops.size();
    size_t matched_op_cnt_required = round(match_coefficient * total_op_cnt);
    for (size_t i = 0; i < total_op_cnt; ++i) {
        if (is_node_to_skip(ordered_ops[i]) &&
            is_node_to_skip(ref_ordered_ops[i]) ||
            m_manager.match(ordered_ops[i], ref_ordered_ops[i])) {
            ++matched_op_cnt;
        }
        if (matched_op_cnt >= matched_op_cnt_required) {
            return true;
        }
    }
    return false;
}

bool
ModelComparator::match(const std::shared_ptr<ov::Model> &model,
                       const std::shared_ptr<ov::Model> &ref,
                       std::map<std::string, InputInfo> &in_info,
                       const std::map<std::string, InputInfo> &in_info_ref) {
    if (match(model, ref)) {
        try {
            in_info = align_input_info(model, ref, in_info, in_info_ref);
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}
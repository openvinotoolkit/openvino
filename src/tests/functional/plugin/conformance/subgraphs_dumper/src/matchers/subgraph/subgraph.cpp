// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include "matchers/subgraph/subgraph.hpp"

using namespace ov::tools::subgraph_dumper;

bool
SubgraphExtractor::match(const std::shared_ptr<ov::Model> &model,
                         const std::shared_ptr<ov::Model> &ref_model) const {
    bool res = comparator.compare(model, ref_model).valid;
    if (res) {
        return res;
    }
    std::vector<std::shared_ptr<ov::Node>> ordered_ops = model->get_ordered_ops(),
                                           ref_ordered_ops = ref_model->get_ordered_ops();
    if (ordered_ops.size() != ref_ordered_ops.size()) {
        return false;
    }
    size_t matched_op_cnt = 0, total_op_cnt = ordered_ops.size();
    size_t matched_op_cnt_required = round(0.9 * total_op_cnt);
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

inline SubgraphExtractor::IsSubgraphTuple prepare_is_subgraph_result(bool is_subgraph,
                                                                     const std::shared_ptr<ov::Model>& graph,
                                                                     const std::shared_ptr<ov::Model>& subgraph,
                                                                     const std::map<std::string, std::string>& matched_ops) {
    return is_subgraph ?
           std::make_tuple(is_subgraph, graph, subgraph, matched_ops) :
           std::make_tuple(is_subgraph, nullptr, nullptr, std::map<std::string, std::string>());
}

SubgraphExtractor::IsSubgraphTuple
SubgraphExtractor::is_subgraph(const std::shared_ptr<ov::Model> &model,
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

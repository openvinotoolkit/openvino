// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

bool
SubgraphExtractor::is_subgraph(const std::shared_ptr<ov::Model> &model,
                               const std::shared_ptr<ov::Model> &ref_model) const {
    std::vector<std::shared_ptr<ov::Node>> ordered_ops = model->get_ordered_ops(),
                                           ref_ordered_ops = ref_model->get_ordered_ops();
    bool is_model = ordered_ops.size() > ref_ordered_ops.size();
    std::vector<std::shared_ptr<ov::Node>> model_to_check_ops, graph_to_check_ops;
    if (is_model) {
        model_to_check_ops = ordered_ops;
        graph_to_check_ops = ref_ordered_ops;
    } else {
        model_to_check_ops = ref_ordered_ops;
        graph_to_check_ops = ordered_ops;
    }
    auto model_it = model_to_check_ops.begin(), graph_it = graph_to_check_ops.begin();
    while (model_it != model_to_check_ops.end() && graph_it != graph_to_check_ops.end()) {
        if (m_manager.match(*model_it, *graph_it)) {
            ++graph_it;
        }
        ++model_it;
    }
    return graph_it == graph_to_check_ops.end();
}
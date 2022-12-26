// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_shape_subgraphs.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/nonconvertible_divide.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;

ov::pass::MarkEntireShapeSubgraphs::MarkEntireShapeSubgraphs() {
    m_markup_func = [](Node* node) {
        ov::disable_fp16_compression(node->shared_from_this());
    };
}

ov::pass::MarkConstantsInShapeSubgraphs::MarkConstantsInShapeSubgraphs() {
    m_markup_func = [](Node* node) {
        if (ov::is_type<ov::opset8::Constant>(node)) {
            ov::disable_fp16_compression(node->shared_from_this());
        }
    };
}

ov::pass::MarkDividesInShapeSubgraphs::MarkDividesInShapeSubgraphs() {
    m_markup_func = [](Node* node) {
        if (ov::is_type<ov::opset8::Divide>(node)) {
            ov::disable_divide_conversion(node->shared_from_this());
        }
    };
}

bool ov::pass::MarkEntireShapeSubgraphs::run_on_model(const shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(MarkEntireShapeSubgraphs);
    deque<Node*> nodes;
    unordered_set<Node*> visited, precision_sensitive_visited;

    for (const auto& r : f->get_results()) {
        nodes.push_back(r.get());
        visited.insert(r.get());
    }
    for (const auto& r : f->get_sinks()) {
        nodes.emplace_back(r.get());
        visited.insert(r.get());
    }

    while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop_front();
        for (const auto& input : curr_node->inputs()) {
            if (ov::is_precision_sensitive(input)) {
                visited.insert(input.get_source_output().get_node());
                // visit_shape_path shouldn't depend on "visited" nodes because we can approach Divide
                // earlier from some non precision sensitive path. So we use dedicated "precision_sensitive_visited"
                // set for precision sensitive nodes, so they can be visited twice and finally marked-up.
                ngraph::op::util::visit_shape_path(input.get_source_output().get_node(),
                                                   precision_sensitive_visited,
                                                   m_markup_func);
            }
        }

        for (auto& input_value : curr_node->input_values()) {
            // continue searching
            const auto& input_node = input_value.get_node();
            if (visited.count(input_node))
                continue;

            if (auto sub_graph_node = ov::as_type<ov::op::util::MultiSubGraphOp>(input_node)) {
                size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
                for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                    auto sub_graph = sub_graph_node->get_function(static_cast<int>(sub_graph_ind));
                    run_on_model(sub_graph);
                }
            }
            nodes.push_front(input_node);
            visited.insert(input_node);
        }
    }
    return true;
}

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_precision_sensitive_subgraphs.hpp"

#include <memory>
#include <vector>

#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;

namespace {
void visit_shape_path(const shared_ptr<ov::Node>& node, unordered_set<shared_ptr<ov::Node>>& visited) {
    if (!node)
        return;
    visited.insert(node);
    deque<shared_ptr<ov::Node>> nodes{node};
    while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop_front();
        // Do not check if already visited
        if (ov::is_type<ov::opset1::ShapeOf>(curr_node) || ov::is_type<ov::opset3::ShapeOf>(curr_node)) {
            continue;
        }
        visited.insert(curr_node);
        if (ov::is_type<ov::opset8::Constant>(curr_node)) {
            ov::disable_fp16_compression(curr_node);
        } else {
            for (auto& input_value : curr_node->input_values()) {
                // continue searching
                const auto& input_node = input_value.get_node_shared_ptr();
                nodes.push_front(input_node);
            }
        }
    }
}
}  // namespace

bool ov::pass::MarkPrecisionSensitiveSubgraphs::run_on_model(const std::shared_ptr<ov::Model>& f) {
    deque<shared_ptr<Node>> nodes;
    unordered_set<shared_ptr<Node>> visited;
    for (auto& r : f->get_results())
        nodes.push_back(r);
    for (auto& r : f->get_sinks())
        nodes.emplace_back(r);

    while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop_front();
        if (visited.count(curr_node))
            continue;
        for (auto& input : curr_node->inputs()) {
            if (ov::is_precision_sensitive(input))
                visit_shape_path(input.get_source_output().get_node_shared_ptr(), visited);
        }
        visited.insert(curr_node);

        for (auto& input_value : curr_node->input_values()) {
            // continue searching
            const auto& input_node = input_value.get_node_shared_ptr();
            nodes.push_front(input_node);
        }
    }
    return true;
}

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detect_causal_mask.hpp"

#include <unordered_set>

#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace {

bool traces_to_range(const std::shared_ptr<ov::Node>& start) {
    std::unordered_set<ov::Node*> visited;
    std::function<bool(const std::shared_ptr<ov::Node>&)> dfs =
        [&](const std::shared_ptr<ov::Node>& n) -> bool {
        if (!n || !visited.insert(n.get()).second)
            return false;
        if (ov::as_type_ptr<ov::op::v4::Range>(n))
            return true;
        if (ov::as_type_ptr<ov::op::v0::Unsqueeze>(n) ||
            ov::as_type_ptr<ov::op::v0::Convert>(n)   ||
            ov::as_type_ptr<ov::op::v0::Squeeze>(n)   ||
            ov::as_type_ptr<ov::op::v1::Reshape>(n))
            return dfs(n->input_value(0).get_node_shared_ptr());
        if (ov::as_type_ptr<ov::op::v1::Add>(n) ||
            ov::as_type_ptr<ov::op::v1::Subtract>(n)) {
            for (size_t i = 0; i < n->get_input_size(); ++i) {
                auto inp = n->input_value(i).get_node_shared_ptr();
                if (!ov::as_type_ptr<ov::op::v0::Constant>(inp) && dfs(inp))
                    return true;
            }
            return false;
        }
        return false;
    };
    return dfs(start);
}

bool is_in_sliding_window_pattern(const std::shared_ptr<ov::Node>& le) {
    for (const auto& out : le->outputs()) {
        for (const auto& consumer_in : out.get_target_inputs()) {
            auto consumer = consumer_in.get_node()->shared_from_this();
            bool is_and = ov::as_type_ptr<ov::op::v13::BitwiseAnd>(consumer) ||
                          ov::as_type_ptr<ov::op::v1::LogicalAnd>(consumer);
            if (!is_and)
                continue;
            for (size_t i = 0; i < consumer->get_input_size(); ++i) {
                auto sibling = consumer->input_value(i).get_node_shared_ptr();
                if (sibling.get() == le.get())
                    continue;
                if (ov::as_type_ptr<ov::op::v1::Greater>(sibling) ||
                    ov::as_type_ptr<ov::op::v1::GreaterEqual>(sibling))
                    return true;
                if (ov::as_type_ptr<ov::op::v13::BitwiseAnd>(sibling)) {
                    for (size_t j = 0; j < sibling->get_input_size(); ++j) {
                        if (ov::as_type_ptr<ov::op::v1::Greater>(
                                sibling->input_value(j).get_node_shared_ptr()) ||
                            ov::as_type_ptr<ov::op::v1::GreaterEqual>(
                                sibling->input_value(j).get_node_shared_ptr()))
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

}  // namespace

namespace ov::npuw {

bool DetectCausalMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op)) {
            if (sdpa->get_causal())
                return true;
            continue;
        }
        auto le = ov::as_type_ptr<ov::op::v1::LessEqual>(op);
        if (!le)
            continue;
        if (!traces_to_range(le->input_value(0).get_node_shared_ptr()))
            continue;
        if (!traces_to_range(le->input_value(1).get_node_shared_ptr()))
            continue;
        if (is_in_sliding_window_pattern(le))
            continue;
        return true;
    }
    return false;
}

}  // namespace ov::npuw

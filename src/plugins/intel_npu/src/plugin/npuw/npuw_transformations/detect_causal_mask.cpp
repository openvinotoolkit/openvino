// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detect_causal_mask.hpp"

#include <cstdlib>
#include <functional>

#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace {

bool traces_to_range(const std::shared_ptr<ov::Node>& start) {
    std::unordered_set<ov::Node*> visited;
    std::function<bool(const std::shared_ptr<ov::Node>&)> dfs = [&](const std::shared_ptr<ov::Node>& n) -> bool {
        if (!n || !visited.insert(n.get()).second)
            return false;
        if (ov::as_type_ptr<ov::op::v4::Range>(n))
            return true;
        if (ov::as_type_ptr<ov::op::v0::Unsqueeze>(n) || ov::as_type_ptr<ov::op::v0::Convert>(n) ||
            ov::as_type_ptr<ov::op::v0::Squeeze>(n) || ov::as_type_ptr<ov::op::v1::Reshape>(n))
            return dfs(n->input_value(0).get_node_shared_ptr());
        if (ov::as_type_ptr<ov::op::v1::Add>(n) || ov::as_type_ptr<ov::op::v1::Subtract>(n)) {
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

bool is_comparison(const std::shared_ptr<ov::Node>& n) {
    return ov::as_type_ptr<ov::op::v1::Greater>(n) || ov::as_type_ptr<ov::op::v1::GreaterEqual>(n) ||
           ov::as_type_ptr<ov::op::v1::Less>(n) || ov::as_type_ptr<ov::op::v1::LessEqual>(n);
}

std::shared_ptr<ov::Node> find_window_bound(const std::shared_ptr<ov::Node>& cmp) {
    for (const auto& out : cmp->outputs()) {
        for (const auto& consumer_in : out.get_target_inputs()) {
            auto consumer = consumer_in.get_node()->shared_from_this();
            bool is_combine = ov::as_type_ptr<ov::op::v13::BitwiseAnd>(consumer) ||
                              ov::as_type_ptr<ov::op::v1::LogicalAnd>(consumer) ||
                              ov::as_type_ptr<ov::op::v13::BitwiseOr>(consumer) ||
                              ov::as_type_ptr<ov::op::v1::LogicalOr>(consumer);
            if (!is_combine)
                continue;
            for (size_t i = 0; i < consumer->get_input_size(); ++i) {
                auto sibling = consumer->input_value(i).get_node_shared_ptr();
                if (sibling.get() == cmp.get())
                    continue;
                if (is_comparison(sibling))
                    return sibling;
                // Phi-3 nests the windowed comparison one level deeper
                if (ov::as_type_ptr<ov::op::v13::BitwiseAnd>(sibling) ||
                    ov::as_type_ptr<ov::op::v13::BitwiseOr>(sibling)) {
                    for (size_t j = 0; j < sibling->get_input_size(); ++j) {
                        auto inner = sibling->input_value(j).get_node_shared_ptr();
                        if (is_comparison(inner))
                            return inner;
                    }
                }
            }
        }
    }
    return nullptr;
}

int64_t extract_window_size_from(const std::shared_ptr<ov::Node>& node) {
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        auto n = node->input_value(i).get_node_shared_ptr();
        while (n && (ov::as_type_ptr<ov::op::v0::Convert>(n) || ov::as_type_ptr<ov::op::v0::Unsqueeze>(n) ||
                     ov::as_type_ptr<ov::op::v0::Squeeze>(n) || ov::as_type_ptr<ov::op::v1::Reshape>(n))) {
            n = n->input_value(0).get_node_shared_ptr();
        }
        if (!n || !(ov::as_type_ptr<ov::op::v1::Add>(n) || ov::as_type_ptr<ov::op::v1::Subtract>(n)))
            continue;
        for (size_t j = 0; j < n->get_input_size(); ++j) {
            auto c = ov::as_type_ptr<ov::op::v0::Constant>(n->input_value(j).get_node_shared_ptr());
            if (!c)
                continue;
            const auto vals = c->cast_vector<int64_t>();
            if (!vals.empty())
                return std::llabs(vals.front());
        }
    }
    return 0;
}

}  // namespace

namespace ov::npuw {

bool DetectAttentionMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_mask_info = MaskInfo{};  // reset to Unknown

    bool found_causal = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op)) {
            if (sdpa->get_causal())
                found_causal = true;
            continue;
        }
        auto le = ov::as_type_ptr<ov::op::v1::LessEqual>(op);
        auto lt = ov::as_type_ptr<ov::op::v1::Less>(op);
        auto cmp = le ? std::static_pointer_cast<ov::Node>(le) : std::static_pointer_cast<ov::Node>(lt);
        if (!cmp)
            continue;
        if (!traces_to_range(cmp->input_value(0).get_node_shared_ptr()))
            continue;
        if (!traces_to_range(cmp->input_value(1).get_node_shared_ptr()))
            continue;
        if (auto window_bound = find_window_bound(cmp)) {
            int64_t window = extract_window_size_from(window_bound);
            if (window == 0)
                window = extract_window_size_from(cmp);
            m_mask_info = {MaskInfo::MaskType::SlidingWindow, window};
            return false;
        }
        found_causal = true;
    }
    m_mask_info = found_causal ? MaskInfo{MaskInfo::MaskType::Causal, 0} : MaskInfo{MaskInfo::MaskType::Unknown, 0};
    return false;
}

}  // namespace ov::npuw

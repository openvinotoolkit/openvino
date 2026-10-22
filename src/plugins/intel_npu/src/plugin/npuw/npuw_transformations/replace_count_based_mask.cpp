// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "replace_count_based_mask.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "../logging.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace {

// Upper bound for the backward walk that proves a bound operand is derived from
// ReduceSum(attention_mask). The real chain is ~10 nodes deep; this only keeps the walk
// from touching the whole model if the pattern check ever lands on an unrelated Select.
constexpr size_t MAX_PROVENANCE_NODES = 256;

// The parameter carrying the attention mask, identified by friendly name or tensor names.
std::shared_ptr<ov::op::v0::Parameter> find_attention_mask(const std::shared_ptr<ov::Model>& model) {
    const std::string key = "attention_mask";
    for (const auto& p : model->get_parameters()) {
        if (p->get_friendly_name().find(key) != std::string::npos) {
            return p;
        }
        for (const auto& name : p->output(0).get_names()) {
            if (name.find(key) != std::string::npos) {
                return p;
            }
        }
    }
    return nullptr;
}

// Last dimension of a [.., T] output, or -1 if not static.
int64_t last_static_dim(const ov::Output<ov::Node>& out) {
    const auto& ps = out.get_partial_shape();
    if (ps.rank().is_dynamic() || ps.rank().get_length() < 1) {
        return -1;
    }
    const auto& d = ps[ps.rank().get_length() - 1];
    return d.is_static() ? d.get_length() : -1;
}

// Matches the "additive mask" idiom: Select(cond, 0.0, -BIG).
bool is_additive_mask_select(const std::shared_ptr<ov::op::v1::Select>& select) {
    auto c_then = ov::as_type_ptr<ov::op::v0::Constant>(select->input_value(1).get_node_shared_ptr());
    auto c_else = ov::as_type_ptr<ov::op::v0::Constant>(select->input_value(2).get_node_shared_ptr());
    if (!c_then || !c_else) {
        return false;
    }
    if (ov::shape_size(c_then->get_shape()) != 1 || ov::shape_size(c_else->get_shape()) != 1) {
        return false;
    }
    const double v_then = c_then->cast_vector<double>().at(0);
    const double v_else = c_else->cast_vector<double>().at(0);
    // "visible" must be 0 and "masked" must be a large negative (-inf, or the dtype's min).
    return v_then == 0.0 && v_else <= -1e4;
}

// The key-index operand: an exclusive CumSum of ones along the LAST (key) axis, which is
// how the exported graph materializes "j = 0..T-1".
bool is_key_index_cumsum(const std::shared_ptr<ov::op::v0::CumSum>& cumsum, int64_t& rank_out) {
    if (!cumsum || !cumsum->is_exclusive() || cumsum->is_reverse()) {
        return false;
    }
    auto axis_c = ov::as_type_ptr<ov::op::v0::Constant>(cumsum->input_value(1).get_node_shared_ptr());
    if (!axis_c || ov::shape_size(axis_c->get_shape()) != 1) {
        return false;
    }
    const auto& ps = cumsum->get_output_partial_shape(0);
    if (ps.rank().is_dynamic()) {
        return false;
    }
    const int64_t rank = ps.rank().get_length();
    if (rank < 2) {
        return false;
    }
    int64_t axis = axis_c->cast_vector<int64_t>().at(0);
    if (axis < 0) {
        axis += rank;
    }
    if (axis != rank - 1) {
        return false;
    }
    rank_out = rank;
    return true;
}

// True if `root` is computed from ReduceSum(attention_mask) - the token COUNT the broken
// idiom is built on. This is what separates this pattern from any other Select(c, 0, -inf):
// without it, a position-based mask could be rewritten too.
bool is_mask_count_derived(const ov::Output<ov::Node>& root, const std::shared_ptr<ov::Node>& mask) {
    std::vector<std::shared_ptr<ov::Node>> stack{root.get_node_shared_ptr()};
    std::unordered_set<ov::Node*> visited;
    while (!stack.empty() && visited.size() < MAX_PROVENANCE_NODES) {
        auto node = stack.back();
        stack.pop_back();
        if (!node || !visited.insert(node.get()).second) {
            continue;
        }
        if (ov::is_type<ov::op::v1::ReduceSum>(node)) {
            // Skip the precision Convert(s) the frontend inserts between the mask and the sum.
            auto src = node->input_value(0).get_node_shared_ptr();
            while (ov::is_type<ov::op::v0::Convert>(src)) {
                src = src->input_value(0).get_node_shared_ptr();
            }
            if (src == mask) {
                return true;
            }
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            stack.push_back(node->input_value(i).get_node_shared_ptr());
        }
    }
    return false;
}

// The rewritten mask is [1,..,1,S,T] where the original was [1,H,S,T], so every consumer
// must broadcast it. In the exported graph the only consumer is the Add that injects the
// mask into the attention scores.
bool consumers_accept_broadcast(const std::shared_ptr<ov::Node>& select) {
    const auto consumers = select->output(0).get_target_inputs();
    if (consumers.empty()) {
        return false;
    }
    for (const auto& input : consumers) {
        auto* eltwise = dynamic_cast<ov::op::util::BinaryElementwiseArithmetic*>(input.get_node());
        if (!eltwise || eltwise->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool ov::npuw::ReplaceCountBasedMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    auto mask_param = find_attention_mask(model);
    if (!mask_param) {
        LOG_DEBUG("ReplaceCountBasedMask: no attention_mask parameter, skipping");
        return false;
    }
    const int64_t mask_len = last_static_dim(mask_param->output(0));
    if (mask_len < 0) {
        LOG_DEBUG("ReplaceCountBasedMask: attention_mask last dim is dynamic, skipping");
        return false;
    }

    // Built lazily on the first match, then reused: every layer gets the very same
    // condition, so partitioning sees one common subexpression instead of 36 copies.
    ov::Output<ov::Node> shared_cond;
    int64_t cached_rank = -1, cached_s = -1, cached_t = -1;

    size_t patched = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto select = ov::as_type_ptr<ov::op::v1::Select>(node);
        if (!select || !is_additive_mask_select(select)) {
            continue;
        }
        auto less = ov::as_type_ptr<ov::op::v1::Less>(select->input_value(0).get_node_shared_ptr());
        if (!less) {
            continue;
        }
        auto cumsum_j = ov::as_type_ptr<ov::op::v0::CumSum>(less->input_value(0).get_node_shared_ptr());
        int64_t rank = 0;
        if (!is_key_index_cumsum(cumsum_j, rank)) {
            continue;
        }
        // Less(key_index, bound): only rewrite when `bound` really is the token count.
        if (!is_mask_count_derived(less->input_value(1), mask_param)) {
            LOG_DEBUG("ReplaceCountBasedMask: " << select->get_friendly_name()
                                                << " is not derived from ReduceSum(attention_mask), skipping");
            continue;
        }

        const auto& ps = cumsum_j->get_output_partial_shape(0);
        if (ps[rank - 1].is_dynamic() || ps[rank - 2].is_dynamic()) {
            LOG_DEBUG("ReplaceCountBasedMask: dynamic S/T on " << select->get_friendly_name() << ", skipping");
            continue;
        }
        const int64_t T = ps[rank - 1].get_length();  // key window
        const int64_t S = ps[rank - 2].get_length();  // query window
        if (S > T) {
            LOG_WARN("ReplaceCountBasedMask: query window " << S << " exceeds key window " << T << " on "
                                                            << select->get_friendly_name() << ", skipping");
            continue;
        }
        // The mask vector must cover the key window, or we cannot index it by j.
        if (mask_len != T) {
            LOG_WARN("ReplaceCountBasedMask: attention_mask length " << mask_len << " != key window " << T << " on "
                                                                     << select->get_friendly_name() << ", skipping");
            continue;
        }
        if (!consumers_accept_broadcast(select)) {
            LOG_DEBUG("ReplaceCountBasedMask: " << select->get_friendly_name()
                                                << " has a consumer that cannot broadcast the mask, skipping");
            continue;
        }
        // All layers share one geometry; a second one would need its own condition.
        if (cached_rank >= 0 && (cached_rank != rank || cached_s != S || cached_t != T)) {
            LOG_WARN("ReplaceCountBasedMask: " << select->get_friendly_name() << " has geometry " << S << "x" << T
                                               << " != " << cached_s << "x" << cached_t << ", skipping");
            continue;
        }

        if (!shared_cond.get_node()) {
            // Query row i owns key position (T - S) + i: the runtime right-aligns the query
            // window inside the key window in prefill, chunked prefill and generate alike.
            const int64_t off = T - S;
            std::vector<char> causal(static_cast<size_t>(S) * static_cast<size_t>(T), 0);
            std::vector<char> diag(static_cast<size_t>(S) * static_cast<size_t>(T), 0);
            for (int64_t i = 0; i < S; ++i) {
                const int64_t i_abs = off + i;
                for (int64_t j = 0; j <= i_abs && j < T; ++j) {
                    causal[static_cast<size_t>(i * T + j)] = 1;
                }
                diag[static_cast<size_t>(i * T + i_abs)] = 1;
            }
            // [1, .., 1, S, T] so it broadcasts against the [B, H, S, T] scores.
            ov::Shape mat_shape(static_cast<size_t>(rank), 1);
            mat_shape[static_cast<size_t>(rank) - 2] = static_cast<size_t>(S);
            mat_shape[static_cast<size_t>(rank) - 1] = static_cast<size_t>(T);
            auto causal_const = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, mat_shape, causal);
            auto diag_const = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, mat_shape, diag);
            causal_const->set_friendly_name("NPUW::ReplaceCountBasedMask/causal");
            diag_const->set_friendly_name("NPUW::ReplaceCountBasedMask/diag");

            // attention_mask -> boolean, shaped [1, .., 1, T] so it broadcasts along rows.
            ov::Shape row_shape(static_cast<size_t>(rank), 1);
            row_shape[static_cast<size_t>(rank) - 1] = static_cast<size_t>(T);
            const std::vector<int64_t> row_dims(row_shape.begin(), row_shape.end());
            auto mask_bool = std::make_shared<ov::op::v0::Convert>(mask_param->output(0), ov::element::boolean);
            auto row_shape_c = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                      ov::Shape{static_cast<size_t>(rank)},
                                                                      row_dims);
            auto mask_row = std::make_shared<ov::op::v1::Reshape>(mask_bool, row_shape_c, false);
            mask_row->set_friendly_name("NPUW::ReplaceCountBasedMask/mask_row");

            // cond = (causal AND attention_mask[j]) OR self
            auto keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_const, mask_row);
            auto cond = std::make_shared<ov::op::v1::LogicalOr>(keep, diag_const);
            keep->set_friendly_name("NPUW::ReplaceCountBasedMask/visible");
            cond->set_friendly_name("NPUW::ReplaceCountBasedMask/cond");

            shared_cond = cond->output(0);
            cached_rank = rank;
            cached_s = S;
            cached_t = T;
        }

        // The old bound chain - including the [1, H, S, T] i32 CumSum - becomes unreachable
        // and drops out of the model with this replacement.
        select->input(0).replace_source_output(shared_cond);
        select->validate_and_infer_types();
        ++patched;
    }

    if (patched == 0) {
        LOG_DEBUG("ReplaceCountBasedMask: no count-based mask found, model left unchanged");
        return false;
    }

    model->validate_nodes_and_infer_types();
    LOG_INFO("ReplaceCountBasedMask: rewrote " << patched << " count-based attention mask(s) to be position-aware");
    return true;
}

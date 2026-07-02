// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_reshape_baked_batch.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/node_registry.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace {

// True if `out` is a Constant holding a single element equal to `value`.
bool is_scalar_constant_with_value(const ov::Output<ov::Node>& out, int64_t value) {
    return ov::op::util::has_constant_value<int64_t>(out.get_node_shared_ptr(), value);
}

// True if `out` is a Constant holding a single positive integer.
bool is_scalar_positive_constant(const ov::Output<ov::Node>& out) {
    int64_t value = 0;
    return ov::op::util::get_constant_value<int64_t>(out.get_node_shared_ptr(), value) && value > 0;
}

// The statically-known last dimension of `ps`, or nullopt when the rank is dynamic, the shape is a
// scalar, or the last dimension itself is dynamic.
std::optional<int64_t> static_last_dim(const ov::PartialShape& ps) {
    if (ps.rank().is_dynamic() || ps.size() == 0)
        return std::nullopt;
    const auto& last = ps[static_cast<std::ptrdiff_t>(ps.size()) - 1];
    if (last.is_static())
        return last.get_length();
    return std::nullopt;
}

// Structural signature of a window-reverse style view whose leading batch was frozen by tracing.
bool passes_structural_gates(const std::shared_ptr<v1::Reshape>& reshape, const std::shared_ptr<v0::Concat>& concat) {
    if (reshape->get_special_zero())
        return false;
    if (concat->get_axis() != 0)
        return false;

    const auto& shape_inputs = concat->input_values();
    if (shape_inputs.size() < 3)
        return false;

    if (!is_scalar_positive_constant(shape_inputs.front()))
        return false;

    const size_t channel_idx = shape_inputs.size() - 1;
    if (!is_scalar_constant_with_value(shape_inputs.back(), -1))
        return false;
    for (size_t i = 0; i + 1 < shape_inputs.size(); ++i) {
        if (is_scalar_constant_with_value(shape_inputs[i], -1))
            return false;  // more than one -1 -> ambiguous, not our pattern
    }

    for (size_t i = 1; i < channel_idx; ++i) {
        if (!ov::as_type_ptr<v0::Constant>(shape_inputs[i].get_node_shared_ptr()))
            return true;
    }
    return false;
}

// Recover the statically-known last (channel) dimension of `data` by walking the graph deterministically.
std::optional<int64_t> resolve_static_last_dim(const ov::Output<ov::Node>& data,
                                               const std::unordered_map<const ov::Node*, int64_t>& pending_channel,
                                               std::unordered_set<const ov::Node*>& visited) {
    const ov::Node* producer = data.get_node();
    if (producer && !visited.insert(producer).second)
        return std::nullopt;

    if (const auto last = static_last_dim(data.get_partial_shape()))
        return last;

    if (auto transpose = ov::as_type_ptr<v1::Transpose>(data.get_node_shared_ptr())) {
        auto order = ov::as_type_ptr<v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
        const auto& in_ps = transpose->input_value(0).get_partial_shape();
        if (order && in_ps.rank().is_static()) {
            const auto perm = order->cast_vector<int64_t>();
            const int64_t rank = in_ps.rank().get_length();
            if (static_cast<int64_t>(perm.size()) == rank && !perm.empty() && perm.back() == rank - 1)
                return resolve_static_last_dim(transpose->input_value(0), pending_channel, visited);
        }
        return std::nullopt;
    }

    if (auto rs = ov::as_type_ptr<v1::Reshape>(data.get_node_shared_ptr())) {
        const auto it = pending_channel.find(rs.get());
        if (it != pending_channel.end())
            return it->second;
    }

    return std::nullopt;
}

// Value-preservation guard for the DIRECT path (channel recovered from the reshape's own static data
// last dim). The rewrite must merely re-partition data's leading dimension and keep data's entire
// trailing block intact — exactly the window-reverse semantics.
bool keeps_data_trailing_block(const std::shared_ptr<v1::Reshape>& reshape,
                               const std::shared_ptr<v0::Concat>& concat,
                               int64_t channel) {
    const auto& data_ps = reshape->input_value(0).get_partial_shape();
    if (data_ps.rank().is_dynamic())
        return false;
    const int64_t rank = data_ps.rank().get_length();
    if (rank < 2)
        return false;

    const auto& shape_inputs = concat->input_values();
    const auto m = static_cast<int64_t>(shape_inputs.size());
    if (m < rank)
        return false;

    for (int64_t j = 1; j < rank; ++j) {
        const auto& data_dim = data_ps[static_cast<std::ptrdiff_t>(j)];
        if (data_dim.is_dynamic())
            return false;
        const auto& shape_elem = shape_inputs[static_cast<size_t>(m - rank + j)];
        if (j == rank - 1) {
            if (channel != data_dim.get_length())
                return false;
        } else if (!is_scalar_constant_with_value(shape_elem, data_dim.get_length())) {
            return false;
        }
    }
    return true;
}

// Rebuild the shape vector for THIS reshape's own data (the concat may be shared between blocks, so we
// must not edit it in place): leading batch -> -1; channel (-1) -> Constant(channel).
void rewrite_reshape(const std::shared_ptr<v1::Reshape>& reshape,
                     const std::shared_ptr<v0::Concat>& concat,
                     int64_t channel) {
    const auto& shape_inputs = concat->input_values();
    const size_t channel_idx = shape_inputs.size() - 1;

    ov::pass::NodeRegistry rg;

    const auto& channel_et = concat->input_value(channel_idx).get_element_type();
    const auto channel_out = rg.make<v0::Constant>(channel_et.is_static() ? channel_et : ov::element::i64,
                                                   ov::Shape{1},
                                                   std::vector<int64_t>{channel});

    const auto& batch_et = shape_inputs.front().get_element_type();
    const auto minus_one = rg.make<v0::Constant>(batch_et.is_static() ? batch_et : ov::element::i64,
                                                 ov::Shape{1},
                                                 std::vector<int64_t>{-1});

    ov::OutputVector new_shape_inputs;
    new_shape_inputs.reserve(shape_inputs.size());
    new_shape_inputs.push_back(minus_one);
    for (size_t i = 1; i < channel_idx; ++i)
        new_shape_inputs.push_back(shape_inputs[i]);
    new_shape_inputs.push_back(channel_out);

    const auto new_concat = rg.make<v0::Concat>(new_shape_inputs, 0);
    reshape->input(1).replace_source_output(new_concat);
    ov::copy_runtime_info(concat, rg.get());
}

}  // namespace

bool ov::pass::RestoreReshapeBakedBatch::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RestoreReshapeBakedBatch);

    struct Candidate {
        std::shared_ptr<v1::Reshape> reshape;
        std::shared_ptr<v0::Concat> concat;
    };

    // Cheap structural scan (NO shape inference): the structural gates inspect only the shape Concat's
    // axis and constant elements — never an inferred PartialShape — so they yield the same verdict before
    // and after inference. We collect the candidates ONCE, here, in get_ordered_ops() topological order
    // (producers before consumers — the walk-back below relies on it).
    std::vector<Candidate> candidates;
    for (const auto& op : model->get_ordered_ops()) {
        auto reshape = ov::as_type_ptr<v1::Reshape>(op);
        if (!reshape)
            continue;
        auto concat = ov::as_type_ptr<v0::Concat>(reshape->input_value(1).get_node_shared_ptr());
        if (!concat)
            continue;
        if (passes_structural_gates(reshape, concat))
            candidates.push_back({reshape, concat});
    }

    // No reshape carries the window-reverse signature (the common case): return without re-inferring shapes.
    if (candidates.empty())
        return false;

    // A candidate exists: materialize the shapes the walk-back keys on. This runs only on the rare model
    // that actually carries the window-reverse signature (guarded by the early return above), so it adds no
    // cost to ordinary models. Validating here keeps the pass self-contained — its channel recovery reads
    // PartialShapes directly and must not depend on the caller's per-pass-validation policy (the sibling
    // dynamic_manager in smart_reshape.cpp turns validation off). validate_nodes_and_infer_types() only
    // re-infers types/shapes — it adds/removes no nodes — so the captured pointers stay valid and the
    // candidate vector stays in topological order.
    model->validate_nodes_and_infer_types();

    struct PendingRewrite {
        std::shared_ptr<v1::Reshape> reshape;
        std::shared_ptr<v0::Concat> concat;
        int64_t channel;
    };
    std::vector<PendingRewrite> pending;
    std::unordered_map<const ov::Node*, int64_t> pending_channel;

    // PHASE 1 — COLLECT (topological order: producers before consumers).
    for (const auto& candidate : candidates) {
        const auto& reshape = candidate.reshape;
        const auto& concat = candidate.concat;

        std::unordered_set<const ov::Node*> visited;
        const auto channel = resolve_static_last_dim(reshape->input_value(0), pending_channel, visited);
        if (!channel)
            continue;

        const auto out_last = static_last_dim(reshape->get_output_partial_shape(0));
        if (out_last && *out_last != *channel)
            continue;

        const bool direct_path = static_last_dim(reshape->input_value(0).get_partial_shape()).has_value();
        if (direct_path && !keeps_data_trailing_block(reshape, concat, *channel))
            continue;

        pending.push_back({reshape, concat, *channel});
        pending_channel.emplace(reshape.get(), *channel);
    }

    // PHASE 2 — REWRITE.
    for (const auto& entry : pending)
        rewrite_reshape(entry.reshape, entry.concat, entry.channel);

    return !pending.empty();
}

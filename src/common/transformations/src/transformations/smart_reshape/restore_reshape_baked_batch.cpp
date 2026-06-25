// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_reshape_baked_batch.hpp"

#include <algorithm>
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

using namespace ov;
using namespace ov::op;

namespace {

// True if `out` is a Constant holding a single element equal to `value`.
bool is_scalar_constant_with_value(const Output<Node>& out, int64_t value) {
    return ov::op::util::has_constant_value<int64_t>(out.get_node_shared_ptr(), value);
}

// True if `out` is a Constant holding a single positive integer.
bool is_scalar_positive_constant(const Output<Node>& out) {
    int64_t value = 0;
    return ov::op::util::get_constant_value<int64_t>(out.get_node_shared_ptr(), value) && value > 0;
}

// The statically-known last dimension of `ps`, or nullopt when the rank is dynamic, the shape is a
// scalar, or the last dimension itself is dynamic.
std::optional<int64_t> static_last_dim(const PartialShape& ps) {
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
std::optional<int64_t> resolve_static_last_dim(const Output<Node>& data,
                                               const std::unordered_map<const Node*, int64_t>& pending_channel,
                                               std::unordered_set<const Node*>& visited) {
    const Node* producer = data.get_node();
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
    const auto channel_out = rg.make<v0::Constant>(channel_et.is_static() ? channel_et : element::i64,
                                                   Shape{1},
                                                   std::vector<int64_t>{channel});

    const auto& batch_et = shape_inputs.front().get_element_type();
    const auto minus_one =
        rg.make<v0::Constant>(batch_et.is_static() ? batch_et : element::i64, Shape{1}, std::vector<int64_t>{-1});

    OutputVector new_shape_inputs;
    new_shape_inputs.reserve(shape_inputs.size());
    new_shape_inputs.push_back(minus_one);
    for (size_t i = 1; i < channel_idx; ++i)
        new_shape_inputs.push_back(shape_inputs[i]);
    new_shape_inputs.push_back(channel_out);

    const auto new_concat = rg.make<v0::Concat>(new_shape_inputs, 0);
    reshape->input(1).replace_source_output(new_concat);
    copy_runtime_info(concat, rg.get());
}

}  // namespace

bool ov::pass::RestoreReshapeBakedBatch::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RestoreReshapeBakedBatch);

    // Cheap structural pre-scan: the structural gates inspect only the shape Concat's axis and constant
    // elements — never an inferred PartialShape — so they need no shape inference. If no reshape carries
    // the window-reverse signature (the common case) we return immediately.
    const auto matches_signature = [](const std::shared_ptr<Node>& op) {
        auto reshape = ov::as_type_ptr<v1::Reshape>(op);
        if (!reshape)
            return false;
        auto concat = ov::as_type_ptr<v0::Concat>(reshape->input_value(1).get_node_shared_ptr());
        return concat && passes_structural_gates(reshape, concat);
    };
    const auto& ordered_ops = model->get_ordered_ops();
    if (std::none_of(ordered_ops.begin(), ordered_ops.end(), matches_signature))
        return false;

    // A candidate exists: make sure the shapes the walk-back keys on are materialized before resolving.
    model->validate_nodes_and_infer_types();

    struct PendingRewrite {
        std::shared_ptr<v1::Reshape> reshape;
        std::shared_ptr<v0::Concat> concat;
        int64_t channel;
    };
    std::vector<PendingRewrite> pending;
    std::unordered_map<const Node*, int64_t> pending_channel;

    // PHASE 1 — COLLECT (topological order: producers before consumers).
    for (const auto& op : ordered_ops) {
        auto reshape = ov::as_type_ptr<v1::Reshape>(op);
        if (!reshape)
            continue;
        auto concat = ov::as_type_ptr<v0::Concat>(reshape->input_value(1).get_node_shared_ptr());
        if (!concat)
            continue;
        if (!passes_structural_gates(reshape, concat))
            continue;

        std::unordered_set<const Node*> visited;
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

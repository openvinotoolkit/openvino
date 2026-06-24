// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_batch_dim_resolver.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/node_registry.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

namespace {
// True if `out` is a Constant holding a single element equal to `value`.
bool is_scalar_constant_with_value(const Output<Node>& out, int64_t value) {
    auto constant = ov::as_type_ptr<v0::Constant>(out.get_node_shared_ptr());
    if (!constant)
        return false;
    if (shape_size(constant->get_shape()) != 1)
        return false;
    const auto values = constant->cast_vector<int64_t>();
    return values.size() == 1 && values[0] == value;
}

// True if `out` is a Constant holding a single positive integer.
bool is_scalar_positive_constant(const Output<Node>& out) {
    auto constant = ov::as_type_ptr<v0::Constant>(out.get_node_shared_ptr());
    if (!constant)
        return false;
    if (shape_size(constant->get_shape()) != 1)
        return false;
    const auto values = constant->cast_vector<int64_t>();
    return values.size() == 1 && values[0] > 0;
}

// Structural signature of a window-reverse style view whose leading batch was frozen by tracing.
// Independent of where the channel value comes from: special_zero false (a 0 already means "copy
// dim", which this rewrite would conflict with); shape concat on axis 0; a leading baked positive-int
// batch constant; exactly one `-1` and it sits in the LAST (channel) position; at least one dynamic
// interior dimension (a genuine shape expression -- a fully-constant shape vector is never a
// propagation bug and would have const-folded away before reaching this pass).
bool passes_structural_gates(const std::shared_ptr<v1::Reshape>& reshape, const std::shared_ptr<v0::Concat>& concat) {
    if (reshape->get_special_zero())
        return false;
    if (concat->get_axis() != 0)
        return false;

    const auto& shape_inputs = concat->input_values();
    // Need a leading batch dim, at least one interior dim, and a trailing channel slot.
    if (shape_inputs.size() < 3)
        return false;

    // Leading element must be a baked positive-int batch constant (e.g. the traced 1).
    if (!is_scalar_positive_constant(shape_inputs.front()))
        return false;

    // The infer slot (-1) must be unique and sit in the LAST (channel) position. This is the
    // window-reverse signature and excludes ordinary reshapes whose -1 is elsewhere/absent.
    const size_t channel_idx = shape_inputs.size() - 1;
    if (!is_scalar_constant_with_value(shape_inputs.back(), -1))
        return false;
    for (size_t i = 0; i + 1 < shape_inputs.size(); ++i) {
        if (is_scalar_constant_with_value(shape_inputs[i], -1))
            return false;  // more than one -1 -> ambiguous, not our pattern
    }

    // At least one interior dimension (between batch and channel) must be dynamic.
    for (size_t i = 1; i < channel_idx; ++i) {
        if (!ov::as_type_ptr<v0::Constant>(shape_inputs[i].get_node_shared_ptr()))
            return true;
    }
    return false;
}

// Recover the statically-known last (channel) dimension of `data` by walking the graph deterministically,
// WITHOUT relying on OV re-inferring/propagating shapes through the intervening permute (which does not
// happen on the real model -- the spatial value-bounds collapse the permuted output to fully dynamic).
//   1. If `data`'s last dimension is already static -> return it. This covers the first window-reverse
//      view (data `[?,8,8,180]`) and any last-dim-preserving eltwise producer in between (e.g. `+ 0.0`),
//      since those keep the static last dimension.
//   2. If the producer is a Transpose whose order keeps the original last axis last (`order.back() ==
//      rank-1`, e.g. `[0,1,3,2,4,5]`) -> recurse into the transposed data: the channel stays trailing.
//   3. If the producer is a Reshape we have already selected for rewrite -> use its resolved channel
//      (this is how the second view resolves through the permute back to the first view's baked channel).
//   4. Otherwise the channel is not statically recoverable -> nullopt (so ordinary `view(1, C, -1)` /
//      attention head-merge, whose data comes from a Parameter with a dynamic last dim, is skipped).
// `visited` guards against re-entry; the graph is a DAG and each step moves strictly to a producer, so
// the walk terminates, but the set keeps it robust to any malformed structure.
std::optional<int64_t> resolve_static_last_dim(const Output<Node>& data,
                                               const std::unordered_map<const Node*, int64_t>& pending_channel,
                                               std::unordered_set<const Node*>& visited) {
    const Node* producer = data.get_node();
    if (producer && !visited.insert(producer).second)
        return std::nullopt;

    // Step 1: a statically-known last dimension on `data` itself.
    const auto& ps = data.get_partial_shape();
    if (ps.rank().is_static() && ps.size() > 0) {
        const auto& last = ps[static_cast<std::ptrdiff_t>(ps.size()) - 1];
        if (last.is_static())
            return last.get_length();
    }

    // Step 2: a last-axis-preserving Transpose -> recurse into the transposed data.
    if (auto transpose = ov::as_type_ptr<v1::Transpose>(data.get_node_shared_ptr())) {
        auto order = ov::as_type_ptr<v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
        const auto& in_ps = transpose->input_value(0).get_partial_shape();
        if (order && in_ps.rank().is_static()) {
            const auto perm = order->cast_vector<int64_t>();
            const int64_t rank = in_ps.rank().get_length();
            // The order must be a full permutation we can reason about, and its last element must keep
            // the original last axis last so the channel remains the trailing dimension.
            if (static_cast<int64_t>(perm.size()) == rank && !perm.empty() && perm.back() == rank - 1)
                return resolve_static_last_dim(transpose->input_value(0), pending_channel, visited);
        }
        return std::nullopt;
    }

    // Step 3: a Reshape already selected for rewrite -> reuse its resolved channel.
    if (auto rs = ov::as_type_ptr<v1::Reshape>(data.get_node_shared_ptr())) {
        const auto it = pending_channel.find(rs.get());
        if (it != pending_channel.end())
            return it->second;
    }

    return std::nullopt;
}

// Value-preservation guard for the DIRECT path (the channel was recovered from the reshape's own data
// last dimension, i.e. data's last dim is static). Pinning the trailing `-1` to data's last dim and
// freeing the leading dim to `-1` is value-preserving ONLY when the rewrite merely re-partitions data's
// leading dimension and keeps data's entire trailing block intact -- exactly the first window-reverse view
// (`data [?,8,8,180]` -> `[B, H//ws, W//ws, 8, 8, 180]`, whose output trailing dims `[8,8,180]` equal data's
// trailing dims). It is NOT value-preserving for ordinary reshapes whose `-1` spans more than data's last
// dim, e.g. a head-merge `Linear(D,D)` then `view(1, T//2, -1)` on data `[?,?,D]` (here `-1 == 2*D != D`):
// those corrupt the result even at the traced batch. We require: data rank >= 2 and static; the shape
// vector has at least one leading dim to absorb the freed batch (size >= data rank); and its trailing
// (rank-1) elements statically equal data's dims [1 .. rank-1] (constant kept dims match data's static
// dims; the trailing `-1` channel matches data's static last dim). When this cannot be proven the rewrite
// is skipped. (The walk-back path -- the second window-reverse view, whose data is the fully dynamic
// permuted output of the first view -- does not take this guard; its channel is resolved structurally
// through a last-axis-preserving transpose to an already-validated upstream view.)
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
    // Need at least one leading dim (beyond the preserved trailing block) to absorb the freed batch.
    if (m < rank)
        return false;

    // The last (rank-1) shape elements must map to data dims [1 .. rank-1]; element at index
    // (m - rank + j) corresponds to data dim j.
    for (int64_t j = 1; j < rank; ++j) {
        const auto& data_dim = data_ps[static_cast<std::ptrdiff_t>(j)];
        if (data_dim.is_dynamic())
            return false;
        const auto& shape_elem = shape_inputs[static_cast<size_t>(m - rank + j)];
        if (j == rank - 1) {
            // The trailing channel slot (the `-1`) is pinned to `channel`; it must equal data's last dim.
            if (channel != data_dim.get_length())
                return false;
        } else if (!is_scalar_constant_with_value(shape_elem, data_dim.get_length())) {
            // A kept interior dim must be a constant equal to data's corresponding (static) dim.
            return false;
        }
    }
    return true;
}

// Rebuild the shape vector for THIS reshape's own data (the concat may be shared between blocks, so we
// must not edit it in place):
//   leading batch -> -1                       (inferred from the real element count)
//   channel (-1)  -> Constant(channel)        (the baked channel recovered by the walk-back)
// Pinning the channel to a static constant (not a runtime Gather) is correct because the channel is
// batch-independent, and it keeps the rewritten reshape's output last dimension static.
void rewrite_reshape(const std::shared_ptr<v1::Reshape>& reshape,
                     const std::shared_ptr<v0::Concat>& concat,
                     int64_t channel) {
    const auto& shape_inputs = concat->input_values();
    const size_t channel_idx = shape_inputs.size() - 1;

    ov::pass::NodeRegistry rg;

    // Keep the same integer element type the original shape elements used.
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
    copy_runtime_info_and_name(concat, rg.get(), {reshape});
}
}  // namespace

bool ReshapeBatchDimResolver::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // Cheap structural pre-scan: the structural gates inspect only the shape Concat's axis and constant
    // elements -- never an inferred PartialShape -- so they need no shape inference. If no reshape carries
    // the window-reverse signature (the common case for the vast majority of converted models) we return
    // immediately, skipping the full-model validation below entirely.
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

    // A candidate exists: make sure the shapes the walk-back keys on (a statically-known data last
    // dimension, e.g. [?,8,8,180]) are materialized before resolving. In the normalize() pipeline the
    // preceding pass already triggers a Validate, but this keeps the pass self-contained and correct if it
    // is ever run standalone or after a non-validating pass.
    model->validate_nodes_and_infer_types();

    struct PendingRewrite {
        std::shared_ptr<v1::Reshape> reshape;
        std::shared_ptr<v0::Concat> concat;
        int64_t channel;
    };
    // Ordered list replayed in phase 2; the map gives the resolver O(1) lookup of an already-selected
    // upstream Reshape's channel (so the second window-reverse view resolves through the permute).
    std::vector<PendingRewrite> pending;
    std::unordered_map<const Node*, int64_t> pending_channel;

    // PHASE 1 -- COLLECT. ordered_ops is topological (producers before consumers), so the first
    // window-reverse view is recorded before the second one is examined, letting the second resolve its
    // channel through the recorded first. validate_nodes_and_infer_types() above only re-inferred shapes
    // on these same nodes; it did not change the node set, so the captured order is still valid.
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

        // Value-preservation guard: if the reshape's inferred output channel is statically known it must
        // equal the recovered channel (the value the rewrite pins `-1` to). If they differ the `-1`
        // resolves to a different product and the rewrite would corrupt the result, so skip it. This is
        // the safety net that closes adversarial cases where a last-axis-preserving walk-back reaches a
        // statically-shaped tensor whose last dim is not the reshape's true `-1` product.
        const auto& out_ps = reshape->get_output_partial_shape(0);
        if (out_ps.rank().is_static() && out_ps.size() > 0) {
            const auto& out_last = out_ps[static_cast<std::ptrdiff_t>(out_ps.size()) - 1];
            if (out_last.is_static() && out_last.get_length() != *channel)
                continue;
        }

        // Stronger guard for the DIRECT path (channel recovered from data's OWN static last dim). When the
        // output last dim is dynamic the cheap guard above is vacuous, so an ordinary reshape whose `-1`
        // spans more than data's last dim (e.g. `Linear(D,D)` then `view(1, T//2, -1)`, `-1 == 2*D`) would
        // slip through and corrupt the result even at the traced batch. Require the rewrite to merely
        // re-partition data's leading dim and keep data's entire trailing block (the genuine window-reverse
        // semantics). The walk-back path (data last dim dynamic -- the second view) is exempt: its channel
        // is resolved structurally through the permute to an already-validated upstream view.
        const auto& data_ps = reshape->input_value(0).get_partial_shape();
        const bool direct_path = data_ps.rank().is_static() && data_ps.size() > 0 &&
                                 data_ps[static_cast<std::ptrdiff_t>(data_ps.size()) - 1].is_static();
        if (direct_path && !keeps_data_trailing_block(reshape, concat, *channel))
            continue;

        pending.push_back({reshape, concat, *channel});
        pending_channel.emplace(reshape.get(), *channel);
    }

    // PHASE 2 -- REWRITE.
    for (const auto& entry : pending)
        rewrite_reshape(entry.reshape, entry.concat, entry.channel);

    return !pending.empty();
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

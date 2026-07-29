// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_reshape_baked_batch.hpp"

#include <memory>
#include <optional>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
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

// True if `transpose` has a constant, full-rank permutation order that keeps the last axis in the last
// position (e.g. [0,1,3,2,4,5]). Such a permute preserves the channel dimension, so the two chained
// window-reverse views share the same (static) channel.
bool is_last_axis_preserving_transpose(const std::shared_ptr<v1::Transpose>& transpose) {
    auto order = ov::as_type_ptr<v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
    const auto& in_ps = transpose->input_value(0).get_partial_shape();
    if (!order || in_ps.rank().is_dynamic())
        return false;
    const auto perm = order->cast_vector<int64_t>();
    const int64_t rank = in_ps.rank().get_length();
    return static_cast<int64_t>(perm.size()) == rank && !perm.empty() && perm.back() == rank - 1;
}

// Value-preservation guard for a reshape whose channel was recovered from its OWN static data last dim.
// The rewrite must merely re-partition the data's leading dimension and keep the data's entire trailing
// block intact — exactly the window-reverse semantics.
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

// The two value-preservation guards for a single reshape given the recovered channel. Guard 1 (output last
// dim, if static, must equal the channel) always applies; guard 2 (trailing-block) applies only when the
// reshape's own data last dim is static.
bool guards_hold(const std::shared_ptr<v1::Reshape>& reshape,
                 const std::shared_ptr<v0::Concat>& concat,
                 int64_t channel) {
    const auto out_last = static_last_dim(reshape->get_output_partial_shape(0));
    if (out_last && *out_last != channel)
        return false;
    const bool direct_path = static_last_dim(reshape->input_value(0).get_partial_shape()).has_value();
    if (direct_path && !keeps_data_trailing_block(reshape, concat, channel))
        return false;
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

ov::pass::RestoreReshapeBakedBatch::RestoreReshapeBakedBatch() {
    MATCHER_SCOPE(RestoreReshapeBakedBatch);

    // Match the exact window-reverse chain: two views separated by a last-axis-preserving Transpose. The
    // shape input of each view is a Concat (ordinary reshapes feed a Constant here). The precise value/
    // arity checks — leading positive-int const, single trailing -1, dynamic interior, and the permute
    // being a full last-axis-preserving order — are done in the callback (wrap_type cannot express them).
    auto inner_concat = pattern::wrap_type<v0::Concat>();
    auto inner_reshape = pattern::wrap_type<v1::Reshape>({pattern::any_input(), inner_concat});
    auto transpose = pattern::wrap_type<v1::Transpose>({inner_reshape, pattern::wrap_type<v0::Constant>()});
    auto outer_concat = pattern::wrap_type<v0::Concat>();
    auto outer_reshape = pattern::wrap_type<v1::Reshape>({transpose, outer_concat});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pm = m.get_pattern_map();
        auto r_in = ov::as_type_ptr<v1::Reshape>(pm.at(inner_reshape));
        auto r_out = ov::as_type_ptr<v1::Reshape>(pm.at(outer_reshape));
        auto c_in = ov::as_type_ptr<v0::Concat>(pm.at(inner_concat));
        auto c_out = ov::as_type_ptr<v0::Concat>(pm.at(outer_concat));
        auto transpose_node = ov::as_type_ptr<v1::Transpose>(pm.at(transpose));
        if (!r_in || !r_out || !c_in || !c_out || !transpose_node)
            return false;

        // Both views must carry the frozen-batch signature.
        if (!passes_structural_gates(r_in, c_in) || !passes_structural_gates(r_out, c_out))
            return false;

        // The permute keeps the last (channel) axis last, so the inner and outer channels are the same.
        if (!is_last_axis_preserving_transpose(transpose_node))
            return false;

        // The channel is the inner view's static data last dim (e.g. 180); it is batch-independent and,
        // through the last-axis-preserving permute, is also the outer view's channel.
        const auto channel = static_last_dim(r_in->input_value(0).get_partial_shape());
        if (!channel)
            return false;

        // Value-preservation guards on both views before any rewrite.
        if (!guards_hold(r_in, c_in, *channel) || !guards_hold(r_out, c_out, *channel))
            return false;

        rewrite_reshape(r_in, c_in, *channel);
        rewrite_reshape(r_out, c_out, *channel);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(outer_reshape, matcher_name);
    register_matcher(m, callback);
}

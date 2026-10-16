// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_get_rows(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    Output<Node> res;
    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    // GGML gathers integer rows as I32 and all floating/quantized rows as F32.
    const auto type = data.get_element_type() == ov::element::i32 ? ov::element::i32 : ov::element::f32;
    const auto convert_rows = [&](ov::Output<Node> rows) {
        if (rows.get_element_type() != type) {
            rows = std::make_shared<ov::op::v0::Convert>(rows, type);
        }
        return rows;
    };

    // The decoder names the gather when it is not ggml's plain row gather:
    //   "gather_elements":     select a column independently in each row (singleton index batches broadcast);
    //   "gather_rows":         gather rows of the token axis (axis 2) with flat indices;
    //   "gather_row_elements": per-row gather over the last axis (MoE gating: probs [1,1,T,E] by selected
    //                          experts [1,1,T,K]), reshaped to [1,T,K,1] for the broadcast against the
    //                          experts' outputs; "shared_row" gathers one shared row (per-expert scales).
    if (context.get_attribute<bool>("gather_elements", false)) {
        auto shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{get_dimensions(data, {0, 1, 2}), get_dimensions(indices, {3})},
            0);
        indices = std::make_shared<ov::op::v3::Broadcast>(indices, shape);
        return rename_outputs_with_suffix(
            {convert_rows(std::make_shared<ov::op::v6::GatherElements>(data, indices, 3))},
            context.get_name());
    }
    if (context.get_attribute<bool>("gather_rows", false)) {
        auto flat_indices =
            std::make_shared<ov::op::v0::Squeeze>(indices,
                                                  ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {2});
        res = std::make_shared<ov::op::v8::Gather>(data, flat_indices, axis);
        return rename_outputs_with_suffix({convert_rows(res)}, context.get_name());
    }
    if (context.get_attribute<bool>("gather_row_elements", false)) {
        // Use an explicit [1,-1,K,1] reshape (K is static, read via PartialShape to avoid .to_shape()
        // throwing when T is dynamic) instead of Squeeze+Unsqueeze, which the CPU plugin implements as a
        // Reshape internally and mis-infers the static pattern when T=1.
        const auto ids_shape = indices.get_partial_shape();
        const int64_t K = ids_shape[ids_shape.size() - 1].get_length();
        auto idx = std::make_shared<ov::op::v0::Convert>(indices, ov::element::i32);
        ov::Output<Node> ge;
        if (context.get_attribute<bool>("shared_row", false)) {
            auto row = std::make_shared<ov::op::v1::Reshape>(data,
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
                                                             false);
            ge = std::make_shared<ov::op::v8::Gather>(row, idx, ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        } else {
            ge = std::make_shared<ov::op::v6::GatherElements>(data, idx, -1);  // [1,1,T,K]
        }
        auto col = std::make_shared<ov::op::v1::Reshape>(
            ge,
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, -1, K, 1}),
            false);  // [1,T,K,1]
        return rename_outputs_with_suffix({convert_rows(col)}, context.get_name());
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    indices =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
    if (data.get_partial_shape().rank() == 4) {
        if (!(data.get_partial_shape()[1].is_dynamic()) && data.get_partial_shape()[1].get_length() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            data = std::make_shared<ov::op::v0::Squeeze>(data,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
        } else {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            data =
                std::make_shared<ov::op::v0::Squeeze>(data, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // A batched gather needs equal batch extents, but a token-independent table tiled to the
            // captured token count (ggml_repeat_4d of a per-expert scale) has a fixed batch while the
            // indices carry the runtime count. Its batch slices are identical, so gather from one slice
            // broadcast to the indices' batch instead of fixing the token count.
            const auto data_ps = data.get_partial_shape();
            const auto idx_ps = indices.get_partial_shape();
            if (data_ps.rank().is_static() && data_ps[0].is_static() && idx_ps.rank().is_static() &&
                idx_ps[0].is_dynamic()) {
                auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto first = std::make_shared<ov::op::v8::Slice>(data, zero, one, one, zero);
                std::vector<int> rest(static_cast<size_t>(data_ps.rank().get_length()) - 1);
                std::iota(rest.begin(), rest.end(), 1);
                auto target = std::make_shared<ov::op::v0::Concat>(
                    OutputVector{get_dimensions(indices, {0}), get_dimensions(first, rest)},
                    0);
                data = std::make_shared<ov::op::v3::Broadcast>(first, target);
            }
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
        }
    } else {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    }

    // The two Squeezes above dropped the leading axes; restore ggml's rank-4 form.
    res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    res = apply_hadamard(context, res);
    return rename_outputs_with_suffix({convert_rows(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op

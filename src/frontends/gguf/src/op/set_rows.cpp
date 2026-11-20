// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// Always emit the internal SetRows placeholder op (never a device-specific ScatterUpdate). A
// normalization-stage lowering replaces it: the built-in LowerSetRows rebuilds the stateless
// ScatterUpdate form; a caller-registered stateful lowering may instead turn the SetRows that
// feeds attention into a stateful KV-cache subgraph. This keeps conversion identical regardless
// of execution mode and needs no KV-vs-non-KV classification at translate time.
namespace {

// Convolution rollback snapshots (llama.cpp build_conv_state_store with n_rs_seq > 0). For n new tokens and
// capacity K, the last Kw = min(n, K) overlapping windows of the [history | tokens] convolution input are stored
// newest first: window w starts at column n - Kw + 1 + w and lands in plane Kw - 1 - w, `snapshot_stride` cache
// rows apart. Inputs: the convolution input [1, seqs, channels, history + n], the slot order (fixed, unused),
// the whole cache [.., rows, row] and the first cache row of the batch.
OutputVector translate_conv_snapshot_set_rows(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto width = context.get_attribute<int64_t>("snapshot_window");
    const auto capacity = context.get_attribute<int64_t>("snapshot_capacity");
    const auto stride = context.get_attribute<int64_t>("snapshot_stride");
    const auto feature_offset = context.get_attribute<int64_t>("feature_offset", 0);

    const auto x = context.get_input(0);
    const auto cache = context.get_input(2);
    const auto x_ps = x.get_partial_shape();
    const auto cache_ps = cache.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(x_ps.rank().is_static() && x_ps.rank().get_length() == 4 && x_ps[2].is_static(),
                                  "Convolution snapshots require a rank-4 input with static channels");
    FRONT_END_OP_CONVERSION_CHECK(cache_ps.rank().is_static() && cache_ps[cache_ps.rank().get_length() - 1].is_static(),
                                  "Convolution snapshots require a static cache row");
    const int64_t window_width = x_ps[2].get_length() * width;
    const int64_t row = cache_ps[cache_ps.rank().get_length() - 1].get_length();
    FRONT_END_OP_CONVERSION_CHECK(feature_offset >= 0 && feature_offset + window_width <= row,
                                  "Convolution snapshot exceeds the cache row");

    const auto i64 = [](std::vector<int64_t> values) {
        return ov::op::v0::Constant::create(ov::element::i64, {values.size()}, values);
    };
    const auto scalar = [](int64_t value) {
        return ov::op::v0::Constant::create(ov::element::i64, {}, {value});
    };
    const auto dim = [&](const ov::Output<ov::Node>& shape, int64_t axis) {
        return std::make_shared<ov::op::v8::Gather>(shape, scalar(axis), scalar(0));
    };

    const auto x_shape = std::make_shared<ov::op::v3::ShapeOf>(x, ov::element::i64);
    const auto tokens = std::make_shared<ov::op::v1::Subtract>(dim(x_shape, 3), scalar(width));
    const auto windows = std::make_shared<ov::op::v1::Minimum>(tokens, scalar(capacity));
    const auto first =
        std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Subtract>(tokens, windows), scalar(1));
    const auto window = std::make_shared<ov::op::v4::Range>(scalar(0), windows, scalar(1), ov::element::i64);

    std::vector<int64_t> offsets(width);
    std::iota(offsets.begin(), offsets.end(), 0);
    const auto columns = std::make_shared<ov::op::v1::Add>(
        std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v0::Unsqueeze>(window, scalar(1)), i64(offsets)),
        first);
    // [1, seqs, channels, Kw, width] -> [Kw * seqs, channels * width]
    ov::Output<ov::Node> values = std::make_shared<ov::op::v8::Gather>(x, columns, scalar(3));
    values = std::make_shared<ov::op::v1::Transpose>(values, i64({3, 0, 1, 2, 4}));
    values = std::make_shared<ov::op::v1::Reshape>(values, i64({-1, window_width}), false);
    if (values.get_element_type() != cache.get_element_type()) {
        values = std::make_shared<ov::op::v0::Convert>(values, cache.get_element_type());
    }

    // rows[w, seq] = slot_begin + seq + (Kw - 1 - w) * stride
    const auto seq = std::make_shared<ov::op::v4::Range>(scalar(0), dim(x_shape, 1), scalar(1), ov::element::i64);
    const auto plane = std::make_shared<ov::op::v1::Multiply>(
        std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Subtract>(windows, scalar(1)), window),
        scalar(stride));
    ov::Output<ov::Node> rows = std::make_shared<ov::op::v1::Add>(
        std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v0::Unsqueeze>(plane, scalar(1)), seq),
        context.get_input(3));
    rows = std::make_shared<ov::op::v1::Reshape>(rows, i64({-1}), false);

    const auto table = std::make_shared<ov::op::v1::Reshape>(cache, i64({-1, row}), false);
    if (feature_offset != 0 || window_width != row) {
        // Keep the rest of each written row
        const auto current = std::make_shared<ov::op::v8::Gather>(table, rows, scalar(0));
        const auto part = [&](int64_t begin, int64_t end) {
            return std::make_shared<ov::op::v8::Slice>(current, i64({begin}), i64({end}), i64({1}), i64({1}));
        };
        values = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{part(0, feature_offset), values, part(feature_offset + window_width, INT64_MAX)},
            1);
    }
    const auto updated = std::make_shared<ov::op::v3::ScatterUpdate>(table, rows, values, scalar(0));
    auto res = std::make_shared<ov::op::v1::Reshape>(updated,
                                                     std::make_shared<ov::op::v3::ShapeOf>(cache, ov::element::i64),
                                                     false);
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

// Rows written into a computed tensor, independently in every [ne3, ne2] batch (e.g. the MoE expert-group
// mask): dst [ne3, ne2, ne1, ne0], data [ne3, ne2, n, ne0], indices [1, ne3, ne2, n] broadcast over batches.
OutputVector translate_batched_set_rows(const NodeContext& context) {
    const auto dst = context.get_input(2);
    ov::Output<ov::Node> data = context.get_input(0);
    if (data.get_element_type() != dst.get_element_type()) {
        data = std::make_shared<ov::op::v0::Convert>(data, dst.get_element_type());
    }
    auto indices = std::make_shared<ov::op::v1::Reshape>(
        context.get_input(1),
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{get_dimensions(context.get_input(1), {1, 2, 3}),
                                                              ov::op::v0::Constant::create(ov::element::i64, {1}, {1})},
                                             0),
        false);
    auto broadcast =
        std::make_shared<ov::op::v3::Broadcast>(indices, std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64));
    auto res =
        std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst,
                                                            broadcast,
                                                            data,
                                                            ov::op::v0::Constant::create(ov::element::i64, {}, {2}));
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace

OutputVector translate_set_rows(const NodeContext& context) {
    if (context.get_attribute<int64_t>("snapshot_window", 0) > 0) {
        return translate_conv_snapshot_set_rows(context);
    }
    num_inputs_check(context, 3, 3);
    if (context.get_attribute<bool>("batched_rows", false)) {
        return translate_batched_set_rows(context);
    }

    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    auto dst = context.get_input(2);

    data = std::make_shared<ov::op::v0::Convert>(data, dst.get_element_type());

    // Row size = the destination cache's innermost dim. Using the dst input (not the SET_ROWS
    // output shape) matters for the flattened KV-cache write (gpt-oss cache_v is stored as
    // [1,1,ctx,1], row_size 1); the output layout can report a different last dim, which would
    // reshape the update to a width the ScatterUpdate dst cannot match.
    const auto dst_ps = context.get_input_shape(2);
    FRONT_END_OP_CONVERSION_CHECK(dst_ps.rank().is_static() && dst_ps[dst_ps.rank().get_length() - 1].is_static(),
                                  "SET_ROWS requires a static destination row size");
    const int64_t row_size = dst_ps[dst_ps.rank().get_length() - 1].get_length();

    auto ind_squeezed =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
    // Flatten the new rows to [.., 1, tokens, row_size]. The leading dim is copied from the incoming
    // data (special_zero) instead of pinned to 1, so the KV write stays in whichever layout the
    // attention block is running in; the stateful lowering re-splits it against the cache shape.
    auto data_reshaped = std::make_shared<ov::op::v1::Reshape>(
        data,
        ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t)0, (int64_t)1, (int64_t)-1, row_size}),
        true);

    auto set_rows = std::make_shared<SetRows>(data_reshaped, ind_squeezed, dst);
    return rename_outputs_with_suffix({std::move(set_rows)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op

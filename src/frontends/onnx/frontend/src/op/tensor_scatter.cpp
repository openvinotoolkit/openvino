// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_24 {

// TensorScatter (ONNX opset 24): a functional KV-cache update.
// Spec: https://onnx.ai/onnx/operators/onnx__TensorScatter.html
//
// present_cache is a copy of past_cache in which, for every batch sample `b` and every
// sequence position `s` of `update`, the slice at the sequence (`axis`) dimension is
// overwritten:
//     present[*, write_indices[b] + s, *] = update[*, s, *]
// For `circular` mode the write position wraps around: (write_indices[b] + s) % max_sequence_length.
//
// The decomposition moves the `axis` dimension next to the batch dimension (position 1) via an
// optional Transpose, then performs the update with a v3::ScatterNDUpdate whose index tensor
// carries (batch, cache_sequence) coordinates for each (b, s) position. All remaining dimensions
// are copied wholesale by ScatterNDUpdate, so their relative order (and the transpose) is irrelevant.
ov::OutputVector tensor_scatter(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node,
                     inputs.size() == 2 || inputs.size() == 3,
                     "TensorScatter expects 2 or 3 inputs (past_cache, update, [write_indices]). Got: ",
                     inputs.size());

    ov::Output<ov::Node> past_cache = inputs[0];
    ov::Output<ov::Node> update = inputs[1];

    const auto rank = past_cache.get_partial_shape().rank();
    CHECK_VALID_NODE(node, rank.is_static(), "TensorScatter requires past_cache to have a static rank.");
    const int64_t r = rank.get_length();

    int64_t axis = node.get_attribute_value<int64_t>("axis", -2);
    if (axis < 0) {
        axis += r;
    }
    CHECK_VALID_NODE(node,
                     axis > 0 && axis < r,
                     "TensorScatter `axis` must select a non-batch dimension within rank bounds. "
                     "Got normalized axis ",
                     axis,
                     " for rank ",
                     r);

    const std::string mode = node.get_attribute_value<std::string>("mode", "linear");
    CHECK_VALID_NODE(node,
                     mode == "linear" || mode == "circular",
                     "TensorScatter supports only `linear` and `circular` modes. Got: ",
                     mode);
    const bool circular = (mode == "circular");

    // The ONNX contract requires sequence_length <= max_sequence_length; the reference
    // implementation rejects violations. When both the cache (`axis`) and update (`axis`)
    // dimensions are static, enforce this at conversion time. Otherwise the dynamic runtime
    // path keeps the behavior defined for valid models; in `circular` mode in particular the
    // modulo must not be relied upon to silently legalize an oversized update.
    const auto past_dim = past_cache.get_partial_shape()[axis];
    const auto update_ps = update.get_partial_shape();
    if (update_ps.rank().is_static() && update_ps.rank().get_length() == r) {
        const auto update_dim = update_ps[axis];
        if (past_dim.is_static() && update_dim.is_static()) {
            CHECK_VALID_NODE(node,
                             update_dim.get_length() <= past_dim.get_length(),
                             "TensorScatter update sequence length (",
                             update_dim.get_length(),
                             ") must not exceed the cache sequence length (",
                             past_dim.get_length(),
                             ").");
        }
    }

    // Move the sequence (`axis`) dimension to position 1 so the scatter index needs only
    // (batch, sequence) coordinates. The swap is its own inverse, reused to transpose back.
    const bool needs_transpose = (axis != 1);
    std::shared_ptr<v0::Constant> perm_const;
    if (needs_transpose) {
        std::vector<int64_t> perm(static_cast<size_t>(r));
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[1], perm[static_cast<size_t>(axis)]);
        perm_const = v0::Constant::create(ov::element::i64, ov::Shape{perm.size()}, perm);
        past_cache = std::make_shared<v1::Transpose>(past_cache, perm_const);
        update = std::make_shared<v1::Transpose>(update, perm_const);
    }

    const auto i64 = ov::element::i64;
    const auto zero = v0::Constant::create(i64, ov::Shape{}, {0});
    const auto one = v0::Constant::create(i64, ov::Shape{}, {1});
    const auto idx_batch_dim = v0::Constant::create(i64, ov::Shape{}, {0});
    const auto idx_seq_dim = v0::Constant::create(i64, ov::Shape{}, {1});
    const auto ax0 = v0::Constant::create(i64, ov::Shape{1}, {0});
    const auto ax1 = v0::Constant::create(i64, ov::Shape{1}, {1});
    const auto ax_last = v0::Constant::create(i64, ov::Shape{1}, {-1});

    // Shapes of the (possibly transposed) tensors: (batch, max_sequence_length, ...rest).
    const auto past_shape = std::make_shared<v3::ShapeOf>(past_cache, i64);
    const auto update_shape = std::make_shared<v3::ShapeOf>(update, i64);
    const auto batch_size = std::make_shared<v8::Gather>(past_shape, idx_batch_dim, zero);        // scalar
    const auto max_seq_len = std::make_shared<v8::Gather>(past_shape, idx_seq_dim, zero);         // scalar
    const auto update_seq_len = std::make_shared<v8::Gather>(update_shape, idx_seq_dim, zero);    // scalar

    // seq_range = [0, 1, ..., sequence_length-1] -> (1, L)
    const auto seq_range = std::make_shared<v4::Range>(zero, update_seq_len, one, i64);
    const auto seq_row = std::make_shared<v0::Unsqueeze>(seq_range, ax0);

    // write offsets per batch sample -> (batch, 1); defaults to zeros when write_indices is absent.
    ov::Output<ov::Node> write_indices;
    if (common::is_input_valid(inputs, 2)) {
        write_indices = std::make_shared<v0::Convert>(inputs[2], i64);
    } else {
        const auto batch_shape = std::make_shared<v0::Unsqueeze>(batch_size, ax0);  // (1,) = [batch]
        write_indices = std::make_shared<v3::Broadcast>(zero, batch_shape);
    }
    // cache_seq[b, s] = write_indices[b] + s, optionally modulo max_sequence_length.
    ov::Output<ov::Node> cache_seq;
    if (circular) {
        // Reduce the offset first, then conditionally wrap before adding. This avoids
        // overflowing int64 when write_indices is large or when the cache length is large.
        const auto normalized_write_indices = std::make_shared<v1::FloorMod>(write_indices, max_seq_len);
        const auto write_col = std::make_shared<v0::Unsqueeze>(normalized_write_indices, ax1);
        const auto distance_to_wrap = std::make_shared<v1::Subtract>(max_seq_len, seq_row);
        const auto wraps = std::make_shared<v1::GreaterEqual>(write_col, distance_to_wrap);
        const auto wrapped_seq = std::make_shared<v1::Subtract>(write_col, distance_to_wrap);
        const auto safe_write_col = std::make_shared<v1::Select>(wraps, zero, write_col);
        const auto unwrapped_seq = std::make_shared<v1::Add>(safe_write_col, seq_row);
        cache_seq = std::make_shared<v1::Select>(wraps, wrapped_seq, unwrapped_seq);
    } else {
        const auto write_col = std::make_shared<v0::Unsqueeze>(write_indices, ax1);
        cache_seq = std::make_shared<v1::Add>(write_col, seq_row);
    }

    // batch coordinate grid -> (batch, L)
    const auto batch_range = std::make_shared<v4::Range>(zero, batch_size, one, i64);
    const auto batch_col = std::make_shared<v0::Unsqueeze>(batch_range, ax1);
    const auto grid_shape = std::make_shared<v3::ShapeOf>(cache_seq, i64);
    const auto batch_grid = std::make_shared<v3::Broadcast>(batch_col, grid_shape);

    // indices[b, s] = [b, cache_seq[b, s]] -> (batch, L, 2)
    const auto batch_idx = std::make_shared<v0::Unsqueeze>(batch_grid, ax_last);
    const auto seq_idx = std::make_shared<v0::Unsqueeze>(cache_seq, ax_last);
    const auto indices = std::make_shared<v0::Concat>(ov::OutputVector{batch_idx, seq_idx}, -1);

    ov::Output<ov::Node> result = std::make_shared<v3::ScatterNDUpdate>(past_cache, indices, update);
    if (needs_transpose) {
        result = std::make_shared<v1::Transpose>(result, perm_const);
    }

    return {result};
}

ONNX_OP("TensorScatter", OPSET_SINCE(24), ai_onnx::opset_24::tensor_scatter);

}  // namespace ov::frontend::onnx::ai_onnx::opset_24

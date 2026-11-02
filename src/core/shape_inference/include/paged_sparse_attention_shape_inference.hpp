// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "openvino/op/paged_sparse_attention.hpp"
#include "utils.hpp"

namespace ov::op::internal {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const PagedSparseAttention* op, const std::vector<TShape>& input_shapes) {
    using TDim = typename TRShape::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 13);
    NODE_VALIDATION_CHECK(op, op->get_key_block_size() > 0);

    const auto& query = input_shapes[0];
    const auto& key = input_shapes[1];
    const auto& value = input_shapes[2];
    const auto& key_cache = input_shapes[3];
    const auto& value_cache = input_shapes[4];
    const auto& past_lens = input_shapes[5];
    const auto& subsequence_begins = input_shapes[6];
    const auto& block_indices = input_shapes[7];
    const auto& block_indices_begins = input_shapes[8];
    const auto& selected_indices = input_shapes[9];
    const auto& selected_count = input_shapes[10];
    const auto& scale = input_shapes[11];
    const auto& sinks = input_shapes[12];

    NODE_SHAPE_INFER_CHECK(op, input_shapes, query.rank().compatible(2));  // `[T, H*S]`
    NODE_SHAPE_INFER_CHECK(op, input_shapes, key.rank().compatible(2));    // `[T, Hkv*S]`
    NODE_SHAPE_INFER_CHECK(op, input_shapes, value.rank().compatible(2));  // `[T, Hkv*Sv]`
    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        key_cache.rank().is_dynamic() ||
            (key_cache.rank().get_length() >= 2 &&
             key_cache.rank().get_length() <= 5));  // `[B, Hkv, Bs, S]`; relaxed for backend-owned storage.
    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        value_cache.rank().is_dynamic() ||
            (value_cache.rank().get_length() >= 2 &&
             value_cache.rank().get_length() <= 5));  // `[B, Hkv, Bs, Sv]`; relaxed for backend-owned storage.
    NODE_SHAPE_INFER_CHECK(op, input_shapes, past_lens.rank().compatible(1));             // `[B_seq]`
    NODE_SHAPE_INFER_CHECK(op, input_shapes, subsequence_begins.rank().compatible(1));    // `[B_seq+1]`
    NODE_SHAPE_INFER_CHECK(op, input_shapes, block_indices.rank().compatible(1));         // `[Nb]`
    NODE_SHAPE_INFER_CHECK(op, input_shapes, block_indices_begins.rank().compatible(1));  // `[B_seq+1]`
    NODE_SHAPE_INFER_CHECK(op, input_shapes, selected_indices.rank().compatible(3));      // `[T, Hs, KB]`
    NODE_SHAPE_INFER_CHECK(op, input_shapes, selected_count.rank().compatible(2));        // `[T, Hs]`
    NODE_SHAPE_INFER_CHECK(op, input_shapes, scale.rank().compatible(0));                 // `[]` (scalar)
    NODE_SHAPE_INFER_CHECK(op, input_shapes, sinks.rank().compatible(1));                 // `[H]` or `[0]`

    const bool query_rank_static = query.rank().is_static();
    const bool key_rank_static = key.rank().is_static();
    const bool value_rank_static = value.rank().is_static();
    const bool selected_indices_rank_static = selected_indices.rank().is_static();
    const bool selected_count_rank_static = selected_count.rank().is_static();

    // `T` must match across query, key, value, selected indices and selected counts.
    auto tokens = query_rank_static ? query[0] : TDim{};
    if (key_rank_static) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(tokens, tokens, key[0]));
    }
    if (value_rank_static) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(tokens, tokens, value[0]));
    }
    if (selected_indices_rank_static) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(tokens, tokens, selected_indices[0]));
    }
    if (selected_count_rank_static) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(tokens, tokens, selected_count[0]));
    }

    // `Hs` must match between selected indices and counts and must be positive.
    auto selection_heads = selected_indices_rank_static ? selected_indices[1] : TDim{};
    if (selected_count_rank_static) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(selection_heads, selection_heads, selected_count[1]));
    }
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !selection_heads.is_static() || selection_heads.get_length() > 0);

    // Both offset arrays have `B_seq+1` entries, with `B_seq` taken from past_lens.
    auto offsets = subsequence_begins.rank().is_static() ? subsequence_begins[0] : TDim{};
    if (block_indices_begins.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(offsets, offsets, block_indices_begins[0]));
    }
    if (past_lens.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(offsets, offsets, past_lens[0] + 1));
    }

    // Logical projection widths `H*S`, `Hkv*S` and `Hkv*Sv` must be positive.
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !query_rank_static || !query[1].is_static() || query[1].get_length() > 0);
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !key_rank_static || !key[1].is_static() || key[1].get_length() > 0);
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !value_rank_static || !value[1].is_static() || value[1].get_length() > 0);

    TDim groups{};
    if (query_rank_static && key_rank_static && key[1].is_static()) {
        // `H/Hkv = query width / key width` must be integral for grouped-query attention.
        groups = query[1] / key[1].get_length();
        check_divided_result(op, groups, query[1], key[1].get_length());
    }
    TDim output_width{};
    if (value_rank_static) {
        // Divide before multiplying: `(H/Hkv) * (Hkv*Sv) = H*Sv`, without an overflowing `Q*V`.
        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            !groups.is_static() || !value[1].is_static() ||
                (groups.get_length() <= ((std::numeric_limits<int64_t>::max() - 1) / value[1].get_length())));
        output_width = groups * value[1];
    }
    // Output is `[T, H*Sv]`; cache layout and logical head counts are validated by the backend.
    return {TRShape{tokens, output_width}};
}

}  // namespace ov::op::internal

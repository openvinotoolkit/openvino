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

    NODE_SHAPE_INFER_CHECK(op, input_shapes, query.rank().compatible(2));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, key.rank().compatible(2));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, value.rank().compatible(2));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, key_cache.rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, value_cache.rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, past_lens.rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, subsequence_begins.rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, block_indices.rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, block_indices_begins.rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, selected_indices.rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, selected_count.rank().compatible(2));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, scale.rank().compatible(0));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, sinks.rank().compatible(1));

    const bool query_rank_static = query.rank().is_static();
    const bool key_rank_static = key.rank().is_static();
    const bool value_rank_static = value.rank().is_static();
    const bool key_cache_rank_static = key_cache.rank().is_static();
    const bool value_cache_rank_static = value_cache.rank().is_static();
    const bool selected_indices_rank_static = selected_indices.rank().is_static();
    const bool selected_count_rank_static = selected_count.rank().is_static();

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

    auto selection_heads = selected_indices_rank_static ? selected_indices[1] : TDim{};
    if (selected_count_rank_static) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(selection_heads, selection_heads, selected_count[1]));
    }
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !selection_heads.is_static() || selection_heads.get_length() > 0);

    auto cache_blocks = key_cache_rank_static ? key_cache[0] : TDim{};
    auto kv_heads = key_cache_rank_static ? key_cache[1] : TDim{};
    auto page_size = key_cache_rank_static ? key_cache[2] : TDim{};
    if (value_cache_rank_static) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(cache_blocks, cache_blocks, value_cache[0]));
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(kv_heads, kv_heads, value_cache[1]));
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(page_size, page_size, value_cache[2]));
    }
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !kv_heads.is_static() || kv_heads.get_length() > 0);
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !page_size.is_static() || page_size.get_length() > 0);
    const auto slots = page_size / op->get_key_block_size();
    check_divided_result(op, slots, page_size, op->get_key_block_size());

    auto offsets = subsequence_begins.rank().is_static() ? subsequence_begins[0] : TDim{};
    if (block_indices_begins.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(offsets, offsets, block_indices_begins[0]));
    }
    if (past_lens.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(offsets, offsets, past_lens[0] + 1));
    }

    auto key_head_size = key_cache_rank_static ? key_cache[3] : TDim{};
    auto value_head_size = value_cache_rank_static ? value_cache[3] : TDim{};
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !key_head_size.is_static() || key_head_size.get_length() > 0);
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !value_head_size.is_static() || value_head_size.get_length() > 0);
    if (key_rank_static && key_head_size.is_static()) {
        const auto projected_kv_heads = key[1] / key_head_size.get_length();
        check_divided_result(op, projected_kv_heads, key[1], key_head_size.get_length());
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(kv_heads, kv_heads, projected_kv_heads));
    }
    if (value_rank_static && value_head_size.is_static()) {
        const auto projected_kv_heads = value[1] / value_head_size.get_length();
        check_divided_result(op, projected_kv_heads, value[1], value_head_size.get_length());
        NODE_SHAPE_INFER_CHECK(op, input_shapes, TDim::merge(kv_heads, kv_heads, projected_kv_heads));
    }
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !kv_heads.is_static() || kv_heads.get_length() > 0);
    if (kv_heads.is_static()) {
        const auto heads = kv_heads.get_length();
        if (key_rank_static) {
            const auto projected_key_head_size = key[1] / heads;
            check_divided_result(op, projected_key_head_size, key[1], heads);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(key_head_size, key_head_size, projected_key_head_size));
        }
        if (value_rank_static) {
            const auto projected_value_head_size = value[1] / heads;
            check_divided_result(op, projected_value_head_size, value[1], heads);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(value_head_size, value_head_size, projected_value_head_size));
        }
    }
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !key_head_size.is_static() || key_head_size.get_length() > 0);
    NODE_SHAPE_INFER_CHECK(op, input_shapes, !value_head_size.is_static() || value_head_size.get_length() > 0);

    TDim query_heads{};
    if (query_rank_static && key_head_size.is_static()) {
        const auto head_size = key_head_size.get_length();
        query_heads = query[1] / head_size;
        check_divided_result(op, query_heads, query[1], head_size);
        NODE_SHAPE_INFER_CHECK(op, input_shapes, !query_heads.is_static() || query_heads.get_length() > 0);
    }
    if (query_heads.is_static()) {
        const auto heads = query_heads.get_length();
        NODE_SHAPE_INFER_CHECK(op, input_shapes, !kv_heads.is_static() || heads % kv_heads.get_length() == 0);
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               !selection_heads.is_static() || heads % selection_heads.get_length() == 0);
        if (sinks.rank().is_static()) {
            NODE_SHAPE_INFER_CHECK(op, input_shapes, sinks[0].compatible(0) || sinks[0].compatible(query_heads));
        }
    }

    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        !query_heads.is_static() || !value_head_size.is_static() ||
            query_heads.get_length() <= std::numeric_limits<int64_t>::max() / value_head_size.get_length());
    return {TRShape{tokens, query_heads * value_head_size}};
}

}  // namespace ov::op::internal

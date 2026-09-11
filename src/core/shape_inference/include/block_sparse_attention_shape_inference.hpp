// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/block_sparse_attention.hpp"
#include "utils.hpp"

namespace ov::op::v17 {

/// \brief Shape inference for BlockSparseAttention.
///
/// Layout (all 4 tensors are rank 4):
///   query:          [B, H,           L, E ]
///   key:            [B, Hk,          S, E ]  (Hk == H, or Hk == 1 broadcast to all heads)
///   value:          [B, Hk,          S, Ev]  (same Hk as key)
///   block_indices:  [B, Hb, L / block_size, k_blocks]  (Hb == H, or Hb == 1 broadcast)
///   block_indices_mask (optional): same shape as block_indices
///   scale (optional): scalar or single-element tensor, not shape-relevant
///   output:         [B, H,           L, Ev]
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const BlockSparseAttention* op, const std::vector<T>& input_shapes) {
    using DimType = typename T::value_type;
    const auto& inputs_count = input_shapes.size();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           inputs_count >= 4 && inputs_count <= 6,
                           "BlockSparseAttention expects 4 to 6 inputs (query, key, value, block_indices, "
                           "[block_indices_mask], [scale]).");
    const bool has_mask = inputs_count >= 5;
    const bool has_scale = inputs_count == 6;

    const auto& query_shape = input_shapes[0];
    const auto& key_shape = input_shapes[1];
    const auto& value_shape = input_shapes[2];
    const auto& block_indices_shape = input_shapes[3];

    for (auto* named_shape : {&query_shape, &key_shape, &value_shape, &block_indices_shape}) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               named_shape->rank().compatible(4),
                               "Query, key, value and block_indices inputs must be rank 4 "
                               "[batch, heads, tokens, features].");
    }
    if (has_mask) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shapes[4].rank().compatible(4),
                               "block_indices_mask must be rank 4, matching block_indices.");
    }

    // Batch and head dims are merged (with broadcasting, like ScaledDotProductAttention) across
    // query/key/value/block_indices/block_indices_mask: each of key/value/block_indices/mask may
    // provide a single shared head (broadcast to all query heads), matching the level of head
    // broadcasting ScaledDotProductAttention itself supports. Arbitrary group sizes (e.g. an
    // integer-multiple GQA grouping where 1 < Hk < H) are intentionally out of scope: real GQA
    // models expand key/value to full head count with Broadcast/Tile before attention, exactly as
    // they do for ScaledDotProductAttention.
    DimType batch_dim{};
    DimType head_dim{};
    DimType l_dim{};
    DimType e_dim{};
    DimType s_dim{};
    DimType ev_dim{};
    DimType num_q_blocks_dim{};
    DimType k_blocks_dim{};

    // NOTE: the accumulator must be *seeded* from the first static-rank shape via plain
    // assignment, not by broadcast_merge-ing from a default-constructed (fully dynamic) DimType:
    // a fully dynamic interval already "contains 1", so broadcast_merge(dynamic, 1) widens back to
    // dynamic instead of narrowing to 1, and the accumulator would never become static.
    bool have_leading_dims = false;
    auto merge_leading_dims = [&](const T& shape) {
        if (!shape.rank().is_static()) {
            return;
        }
        if (!have_leading_dims) {
            batch_dim = shape[0];
            head_dim = shape[1];
            have_leading_dims = true;
            return;
        }
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::broadcast_merge(batch_dim, batch_dim, shape[0]) &&
                                   DimType::broadcast_merge(head_dim, head_dim, shape[1]),
                               "Incompatible batch/heads dimensions between query, key, value, "
                               "block_indices and block_indices_mask.");
    };
    merge_leading_dims(query_shape);
    merge_leading_dims(key_shape);
    merge_leading_dims(value_shape);
    merge_leading_dims(block_indices_shape);
    if (has_mask) {
        merge_leading_dims(input_shapes[4]);
    }

    if (query_shape.rank().is_static()) {
        l_dim = query_shape[2];
        e_dim = query_shape[3];
    }
    if (key_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(e_dim, e_dim, key_shape[3]),
                               "The head size (last dimension) of query and key must match.");
        s_dim = key_shape[2];
    }
    if (value_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(s_dim, s_dim, value_shape[2]),
                               "The number of tokens (second-to-last dimension) of key and value must match.");
        ev_dim = value_shape[3];
    }
    if (block_indices_shape.rank().is_static()) {
        num_q_blocks_dim = block_indices_shape[2];
        k_blocks_dim = block_indices_shape[3];
    }
    if (has_mask && input_shapes[4].rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(num_q_blocks_dim, num_q_blocks_dim, input_shapes[4][2]) &&
                                   DimType::merge(k_blocks_dim, k_blocks_dim, input_shapes[4][3]),
                               "block_indices_mask shape must match block_indices shape.");
    }
    if (has_scale) {
        const auto& scale_shape = input_shapes[5];
        if (scale_shape.rank().is_static()) {
            const bool scale_is_scalar = scale_shape.rank().compatible(0);
            const bool scale_has_one_elem = scale_shape.rank().compatible(1) && scale_shape[0].compatible(1);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   scale_is_scalar || scale_has_one_elem,
                                   "The scale input must be scalar or have a single element.");
        }
    }

    const auto& block_size = op->get_block_size();
    if (block_size > 0) {
        if (l_dim.is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   l_dim.get_length() % block_size == 0,
                                   "The query length must be a multiple of 'block_size'.");
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   DimType::merge(num_q_blocks_dim,
                                                  num_q_blocks_dim,
                                                  DimType(l_dim.get_length() / block_size)),
                                   "block_indices' num_query_blocks dimension must equal query_len / block_size.");
        }
        if (s_dim.is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   s_dim.get_length() % block_size == 0,
                                   "The key/value length must be a multiple of 'block_size'.");
        }
    }

    auto output_shapes = std::vector<TRShape>{TRShape{}};
    auto& out = output_shapes[0];
    out.resize(4);
    out[0] = batch_dim;
    out[1] = head_dim;
    out[2] = l_dim;
    out[3] = ev_dim;
    return output_shapes;
}

}  // namespace ov::op::v17

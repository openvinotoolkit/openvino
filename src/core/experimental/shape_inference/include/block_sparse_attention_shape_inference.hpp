// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/experimental/block_sparse_attention.hpp"
#include "utils.hpp"

namespace ov::op::experimental {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const BlockSparseAttention* op, const std::vector<T>& input_shapes) {
    using DimType = typename T::value_type;
    const auto& input_size = input_shapes.size();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_size == 4 || input_size == 5,
                           "BlockSparseAttention expects 4 or 5 inputs.");

    const auto& q_shape = input_shapes[0];
    const auto& k_shape = input_shapes[1];
    const auto& v_shape = input_shapes[2];
    const auto& mask_shape = input_shapes[3];

    NODE_SHAPE_INFER_CHECK(op, input_shapes, q_shape.rank().compatible(4), "query must be rank 4: [B, Lq, Hq, E].");
    NODE_SHAPE_INFER_CHECK(op, input_shapes, k_shape.rank().compatible(4), "key must be rank 4: [B, Lk, Hkv, E].");
    NODE_SHAPE_INFER_CHECK(op, input_shapes, v_shape.rank().compatible(4), "value must be rank 4: [B, Lk, Hkv, Ev].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           mask_shape.rank().compatible(4),
                           "block_mask must be rank 4: [Bm, Hm, num_q_blocks, num_kv_blocks].");

    const auto block_size_q = op->get_block_size_q();
    const auto block_size_kv = op->get_block_size_kv();

    DimType b_dim{}, lq_dim{}, lk_dim{}, hq_dim{}, hkv_dim{}, e_dim{}, ev_dim{};

    if (q_shape.rank().is_static()) {
        b_dim = q_shape[0];
        lq_dim = q_shape[1];
        hq_dim = q_shape[2];
        e_dim = q_shape[3];
    }
    if (k_shape.rank().is_static()) {
        // Key/value batch is not broadcastable against query (unlike block_mask's Bm below):
        // evaluate() and the reference/GPU kernels index key/value with the query's own batch
        // index directly, with no broadcast fallback, so a mismatched-but-broadcast-compatible
        // (e.g. 1 vs B) key/value batch would read out of bounds. Use a strict merge.
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(b_dim, b_dim, k_shape[0]),
                               "Batch dim of key is not compatible with query.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(hkv_dim, hkv_dim, k_shape[2]),
                               "Hkv dim of key is not compatible with value.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(e_dim, e_dim, k_shape[3]),
                               "Feature dim (E) of key is not compatible with query.");
        lk_dim = k_shape[1];
    }
    if (v_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(b_dim, b_dim, v_shape[0]),
                               "Batch dim of value is not compatible with query/key.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(lk_dim, lk_dim, v_shape[1]),
                               "Lk dim of value is not compatible with key.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               DimType::merge(hkv_dim, hkv_dim, v_shape[2]),
                               "Hkv dim of value is not compatible with key.");
        ev_dim = v_shape[3];
    }

    if (hq_dim.is_static() && hkv_dim.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               hkv_dim.get_length() > 0 && hq_dim.get_length() % hkv_dim.get_length() == 0,
                               "Hq (",
                               hq_dim,
                               ") must be evenly divisible by Hkv (",
                               hkv_dim,
                               ").");
    }

    if (mask_shape.rank().is_static()) {
        // block_mask's Bm genuinely broadcasts (unlike key/value's batch above): the reference
        // explicitly indexes it as `(Bm == 1) ? 0 : b`. Validate compatibility without
        // merging it into b_dim -- the output batch is derived from query/key/value alone (see
        // the `b_dim` used to build output_shape below), matching evaluate()'s `B = q_shape[0]`.
        const auto& bm_dim = mask_shape[0];
        if (bm_dim.is_static() && b_dim.is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   bm_dim.get_length() == 1 || bm_dim.get_length() == b_dim.get_length(),
                                   "block_mask batch dim (",
                                   bm_dim,
                                   ") must be 1 or equal to B (",
                                   b_dim,
                                   ").");
        }

        const auto& hm_dim = mask_shape[1];
        if (hm_dim.is_static() && (hq_dim.is_static() || hkv_dim.is_static())) {
            const auto hm = hm_dim.get_length();
            const bool matches_hkv = hkv_dim.is_static() && hm == hkv_dim.get_length();
            const bool matches_hq = hq_dim.is_static() && hm == hq_dim.get_length();
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   hm == 1 || matches_hkv || matches_hq,
                                   "block_mask head dim (",
                                   hm_dim,
                                   ") must be 1, Hkv, or Hq.");
        }

        if (mask_shape[2].is_static() && lq_dim.is_static()) {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                mask_shape[2].get_length() == (lq_dim.get_length() + block_size_q - 1) / block_size_q,
                "block_mask query-block dim must equal ceil(Lq / block_size_q).");
        }
        if (mask_shape[3].is_static() && lk_dim.is_static()) {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                mask_shape[3].get_length() == (lk_dim.get_length() + block_size_kv - 1) / block_size_kv,
                "block_mask key-block dim must equal ceil(Lk / block_size_kv).");
        }
    }

    auto output_shape = TRShape{b_dim, lq_dim, hq_dim, ev_dim};
    return {output_shape};
}

}  // namespace ov::op::experimental
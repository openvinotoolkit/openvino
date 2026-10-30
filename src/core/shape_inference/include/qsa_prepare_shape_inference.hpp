// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "utils.hpp"

namespace ov::op::v17 {
/// \brief Checks whether the optional input of QSAPrepare at the given port is disabled.
///
/// An optional input is disabled by an empty Constant.
inline bool qsa_prepare_input_disabled(const Node* op, size_t port) {
    return port < op->get_input_size() && ov::util::is_empty_constant_tensor(op->input_value(port));
}

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const QSAPrepare* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TShape::value_type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 8);

    const auto key_block_size = op->get_key_block_size();
    NODE_VALIDATION_CHECK(op,
                          key_block_size > 1,
                          "The attribute key_block_size must be an integer greater than 1. Got: ",
                          key_block_size);

    const auto& raw_index_keys_shape = input_shapes[0];
    const auto& past_block_index_keys_shape = input_shapes[1];
    const auto& key_length_shape = input_shapes[2];
    const auto& rope_cos_shape = input_shapes[3];
    const auto& rope_sin_shape = input_shapes[4];
    const auto& k_norm_weight_shape = input_shapes[5];
    const auto& past_incomplete_raw_keys_shape = input_shapes[6];
    const auto& past_incomplete_rope_shape = input_shapes[7];

    const auto k_norm_weight_disabled = qsa_prepare_input_disabled(op, 5);
    const auto rope_disabled = qsa_prepare_input_disabled(op, 3);
    const auto is_raw_keys_rank_static = raw_index_keys_shape.rank().is_static();

    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        qsa_prepare_input_disabled(op, 4) == rope_disabled && qsa_prepare_input_disabled(op, 7) == rope_disabled,
        "The inputs rope_cos, rope_sin and past_incomplete_rope must be empty constants "
        "together to disable the rotation.");

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           raw_index_keys_shape.rank().compatible(3),
                           "The shape of raw_index_keys must be 3D [N, L, Di].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           past_block_index_keys_shape.rank().compatible(3),
                           "The shape of past_block_index_keys must be 3D [N, Bp, Di].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           key_length_shape.rank().compatible(1),
                           "The shape of key_length must be 1D [N] or [1].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           k_norm_weight_disabled || k_norm_weight_shape.rank().compatible(1),
                           "The shape of k_norm_weight must be 1D [Di], or an empty constant to disable "
                           "the normalization.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           past_incomplete_raw_keys_shape.rank().compatible(3),
                           "The shape of past_incomplete_raw_keys must be 3D [N, key_block_size - 1, Di].");
    if (!rope_disabled) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               rope_cos_shape.rank().compatible(3) && rope_sin_shape.rank().compatible(3),
                               "The shape of rope_cos and rope_sin must be 3D [N, L, Dr].");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               past_incomplete_rope_shape.rank().compatible(2),
                               "The shape of past_incomplete_rope must be 2D [N, 2 * Dr].");
    }

    TDim batch{};
    TDim head_size{};
    if (is_raw_keys_rank_static) {
        batch = raw_index_keys_shape[0];
        head_size = raw_index_keys_shape[2];
    }

    if (past_block_index_keys_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(batch, batch, past_block_index_keys_shape[0]),
                               "The batch dimension of raw_index_keys and past_block_index_keys must be the same.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(head_size, head_size, past_block_index_keys_shape[2]),
                               "The head size of raw_index_keys and past_block_index_keys must be the same.");
    }

    if (past_incomplete_raw_keys_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(batch, batch, past_incomplete_raw_keys_shape[0]),
                               "The batch dimension of raw_index_keys and past_incomplete_raw_keys must be the same.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(head_size, head_size, past_incomplete_raw_keys_shape[2]),
                               "The head size of raw_index_keys and past_incomplete_raw_keys must be the same.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               past_incomplete_raw_keys_shape[1].compatible(key_block_size - 1),
                               "The second dimension of past_incomplete_raw_keys must be key_block_size - 1, "
                               "expected: ",
                               key_block_size - 1,
                               ", got: ",
                               past_incomplete_raw_keys_shape[1]);
    }

    if (!k_norm_weight_disabled && k_norm_weight_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(head_size, head_size, k_norm_weight_shape[0]),
                               "The size of k_norm_weight must be equal to the head size of raw_index_keys.");
    }

    // Rotated dimensions, doubled by the cos / sin packing of the incomplete block RoPE row.
    auto rotated_size = rope_disabled ? TDim(0) : TDim(ov::util::dim::inf_bound);
    if (!rope_disabled && rope_cos_shape.rank().is_static() && rope_sin_shape.rank().is_static()) {
        TDim merged;
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(merged, rope_cos_shape[2], rope_sin_shape[2]),
                               "The number of rotated dimensions of rope_cos and rope_sin must be the same.");
        if (merged.is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   merged.get_length() > 0 && merged.get_length() % 2 == 0,
                                   "The number of rotated dimensions must be a positive even integer. Got: ",
                                   merged);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   head_size.is_dynamic() || merged.get_length() <= head_size.get_length(),
                                   "The number of rotated dimensions must not exceed the head size of "
                                   "raw_index_keys.");
        }
        rotated_size = merged + merged;
        // The RoPE inputs broadcast over the batch, so the dimension is checked, not merged into it.
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               rope_cos_shape[0].compatible(1) || rope_cos_shape[0].compatible(batch),
                               "The batch dimension of rope_cos must be 1 or the batch of raw_index_keys.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               rope_sin_shape[0].compatible(1) || rope_sin_shape[0].compatible(batch),
                               "The batch dimension of rope_sin must be 1 or the batch of raw_index_keys.");
        if (is_raw_keys_rank_static) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(merged, rope_cos_shape[1], raw_index_keys_shape[1]),
                                   "The number of new tokens of rope_cos and raw_index_keys must be the same.");
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(merged, rope_sin_shape[1], raw_index_keys_shape[1]),
                                   "The number of new tokens of rope_sin and raw_index_keys must be the same.");
        }
        if (past_incomplete_rope_shape.rank().is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(rotated_size, rotated_size, past_incomplete_rope_shape[1]),
                                   "The second dimension of past_incomplete_rope must be 2 * Dr.");
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(batch, batch, past_incomplete_rope_shape[0]),
                                   "The batch dimension of raw_index_keys and past_incomplete_rope must be the same.");
        }
    }

    // The number of complete blocks of the longest sequence is known only from the key_length values.
    auto num_blocks = TDim(ov::util::dim::inf_bound);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           key_length_shape.rank().is_dynamic() || key_length_shape[0].compatible(1) ||
                               key_length_shape[0].compatible(batch),
                           "The size of key_length must be 1 or the batch of raw_index_keys.");
    if (const auto key_lengths = get_input_const_data_as<TRShape, int64_t>(op, 2, ta)) {
        int64_t longest = 0;
        for (const auto& length : *key_lengths) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   length >= 0,
                                   "The key_length values must be non-negative. Got: ",
                                   length);
            const auto blocks = length / key_block_size;
            longest = blocks > longest ? blocks : longest;
        }
        num_blocks = TDim(longest);

        if (is_raw_keys_rank_static && raw_index_keys_shape[1].is_static() &&
            past_block_index_keys_shape.rank().is_static() && past_block_index_keys_shape[1].is_static()) {
            const auto new_tokens = raw_index_keys_shape[1].get_length();
            int64_t cached = 0;
            for (const auto& length : *key_lengths) {
                const auto past = length - new_tokens;
                const auto blocks = past > 0 ? past / key_block_size : 0;
                cached = blocks > cached ? blocks : cached;
            }
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   past_block_index_keys_shape[1].get_length() >= cached,
                                   "The second dimension of past_block_index_keys must cover the complete "
                                   "blocks of the past tokens, at least: ",
                                   cached,
                                   ", got: ",
                                   past_block_index_keys_shape[1]);
        }
    } else if (is_raw_keys_rank_static && past_block_index_keys_shape.rank().is_static()) {
        // Without the key_length values only the upper bound is known: the cached blocks plus the new tokens,
        // as the past incomplete block contributes at most key_block_size - 1 tokens.
        const auto upper =
            past_block_index_keys_shape[1] +
            ov::util::dim::ceil_div(raw_index_keys_shape[1], static_cast<typename TDim::value_type>(key_block_size));
        if (upper.is_static()) {
            num_blocks = TDim(0, upper.get_length());
        }
    }

    return {TRShape{batch, std::move(num_blocks), head_size},
            TRShape{batch, TDim(key_block_size - 1), head_size},
            TRShape{batch, std::move(rotated_size)}};
}
}  // namespace ov::op::v17

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/sparse_attention.hpp"
#include "openvino/util/math_util.hpp"
#include "utils.hpp"

namespace ov::op::v17 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SparseAttention* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TShape::value_type;

    const auto inputs_count = input_shapes.size();
    NODE_VALIDATION_CHECK(op, 5 <= inputs_count && inputs_count <= 8);

    const auto key_block_size = op->get_key_block_size();
    const auto query_block_size = op->get_query_block_size();
    NODE_VALIDATION_CHECK(op,
                          key_block_size > 0,
                          "The attribute key_block_size must be a positive integer. Got: ",
                          key_block_size);
    NODE_VALIDATION_CHECK(op,
                          query_block_size > 0,
                          "The attribute query_block_size must be a positive integer. Got: ",
                          query_block_size);

    const auto& query_shape = input_shapes[0];
    const auto& key_shape = input_shapes[1];
    const auto& value_shape = input_shapes[2];
    const auto& block_indices_shape = input_shapes[3];
    const auto& block_count_shape = input_shapes[4];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           query_shape.rank().compatible(4),
                           "The shape of query must be 4D [N, H, L, E].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           key_shape.rank().compatible(4),
                           "The shape of key must be 4D [N, Hkv, S, E].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           value_shape.rank().compatible(4),
                           "The shape of value must be 4D [N, Hkv, S, Ev].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           block_indices_shape.rank().compatible(4),
                           "The shape of block_indices must be 4D [N, Hs, Lq, KB].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           block_count_shape.rank().compatible(3),
                           "The shape of block_count must be 3D [N, Hs, Lq].");

    // Output is [N, H, L, Ev]: the query shape with the value head size as the last dimension.
    auto output_shapes = std::vector<TRShape>{input_shapes[0]};
    auto& out_shape = output_shapes[0];
    const auto is_out_rank_static = out_shape.rank().is_static();

    // Batch size, shared by all the data and selection inputs
    if (is_out_rank_static) {
        for (size_t i = 1; i < 5; ++i) {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                input_shapes[i].rank().is_dynamic() || TDim::merge(out_shape[0], out_shape[0], input_shapes[i][0]),
                "The batch dimension of all the inputs must be the same.");
        }
    }

    if (key_shape.rank().is_static() && value_shape.rank().is_static()) {
        TDim merged;
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(merged, key_shape[0], value_shape[0]),
                               "The batch dimension of all the inputs must be the same.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(merged, key_shape[1], value_shape[1]),
                               "The number of key and value heads must be the same.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(merged, key_shape[2], value_shape[2]),
                               "The number of key and value positions must be the same.");
    }

    if (query_shape.rank().is_static() && key_shape.rank().is_static()) {
        TDim merged;
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(merged, query_shape[3], key_shape[3]),
                               "The head size of query and key must be the same.");
        if (query_shape[1].is_static() && key_shape[1].is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   key_shape[1].get_length() > 0,
                                   "The number of key and value heads must be a positive integer. Got: ",
                                   key_shape[1]);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   query_shape[1].get_length() % key_shape[1].get_length() == 0,
                                   "The number of query heads must be divisible by the number of key heads.");
        }
        if (op->get_causal() && query_shape[2].is_static() && key_shape[2].is_static()) {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                query_shape[2].get_length() <= key_shape[2].get_length(),
                "In the causal mode the number of query tokens must not exceed the number of key positions.");
        }
    }

    TDim selection_heads{};
    TDim query_rows{};
    if (block_indices_shape.rank().is_static()) {
        selection_heads = block_indices_shape[1];
        query_rows = block_indices_shape[2];
    }

    if (block_count_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(selection_heads, selection_heads, block_count_shape[1]),
                               "The number of selection heads of block_indices and block_count must be the same.");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(query_rows, query_rows, block_count_shape[2]),
                               "The number of query rows of block_indices and block_count must be the same.");
        if (block_indices_shape.rank().is_static()) {
            TDim merged;
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(merged, block_indices_shape[0], block_count_shape[0]),
                                   "The batch dimension of all the inputs must be the same.");
        }
    }

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           selection_heads.is_dynamic() || selection_heads.get_length() > 0,
                           "The number of selection heads must be a positive integer. Got: ",
                           selection_heads);

    if (query_shape.rank().is_static()) {
        if (query_shape[1].is_static() && selection_heads.is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   query_shape[1].get_length() % selection_heads.get_length() == 0,
                                   "The number of query heads must be divisible by the number of selection heads.");
        }
        if (query_shape[2].is_static() && query_rows.is_static()) {
            const auto expected_rows = ov::util::ceil_div(query_shape[2].get_length(), query_block_size);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   query_rows.get_length() == expected_rows,
                                   "The number of query rows must be ceil(L / query_block_size), expected: ",
                                   expected_rows,
                                   ", got: ",
                                   query_rows);
        }
    }

    if (inputs_count > 5) {
        const auto& key_length_shape = input_shapes[5];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               key_length_shape.rank().compatible(1),
                               "The shape of key_length must be 1D [N] or [1].");
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               !is_out_rank_static || key_length_shape.rank().is_dynamic() ||
                                   key_length_shape[0].compatible(1) || key_length_shape[0].compatible(out_shape[0]),
                               "The size of key_length must be 1 or the batch dimension of the other inputs.");
        if (key_shape.rank().is_static() && key_shape[2].is_static()) {
            if (const auto key_lengths = get_input_const_data_as<TRShape, int64_t>(op, 5, ta)) {
                const auto num_positions = static_cast<int64_t>(key_shape[2].get_length());
                for (const auto& length : *key_lengths) {
                    NODE_SHAPE_INFER_CHECK(op,
                                           input_shapes,
                                           0 <= length && length <= num_positions,
                                           "The key_length values must be in [0, S], S = ",
                                           num_positions,
                                           ". Got: ",
                                           length);
                }
            }
        }
    }

    if (inputs_count > 6) {
        const auto& scale_shape = input_shapes[6];
        const auto& scale_rank = scale_shape.rank();
        if (scale_rank.is_static()) {
            const auto scale_is_scalar = scale_rank.get_length() == 0;
            const auto scale_has_one_elem = scale_rank.get_length() == 1 && scale_shape[0].compatible(1);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   scale_is_scalar || scale_has_one_elem,
                                   "The scale input must be a scalar or a single element 1D tensor.");
        }
    }

    if (inputs_count > 7) {
        const auto& sinks_shape = input_shapes[7];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               sinks_shape.rank().compatible(1),
                               "The shape of sinks must be 1D [H].");
        if (sinks_shape.rank().is_static() && query_shape.rank().is_static()) {
            TDim merged;
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(merged, sinks_shape[0], query_shape[1]),
                                   "The number of sinks must be equal to the number of query heads.");
        }
    }

    TDim value_head_size{};
    if (value_shape.rank().is_static()) {
        value_head_size = value_shape[3];
    }
    if (is_out_rank_static) {
        out_shape.resize(3);
        out_shape.push_back(std::move(value_head_size));
    }
    return output_shapes;
}
}  // namespace ov::op::v17

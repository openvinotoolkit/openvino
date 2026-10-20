// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/sparse_attention_indexer.hpp"
#include "utils.hpp"

namespace ov::op::v17 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SparseAttentionIndexer* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TShape::value_type;

    const auto inputs_count = input_shapes.size();
    NODE_VALIDATION_CHECK(op, inputs_count == 3 || inputs_count == 4);

    const auto key_block_size = op->get_key_block_size();
    const auto token_budget = op->get_token_budget();
    const auto num_selection_heads = op->get_num_selection_heads();
    NODE_VALIDATION_CHECK(op,
                          key_block_size > 0,
                          "The attribute key_block_size must be a positive integer. Got: ",
                          key_block_size);
    NODE_VALIDATION_CHECK(op,
                          token_budget > 0,
                          "The attribute token_budget must be a positive integer. Got: ",
                          token_budget);
    NODE_VALIDATION_CHECK(op,
                          token_budget % key_block_size == 0,
                          "The attribute token_budget must be divisible by key_block_size. Got: ",
                          token_budget,
                          " and ",
                          key_block_size);
    NODE_VALIDATION_CHECK(op,
                          num_selection_heads > 0,
                          "The attribute num_selection_heads must be a positive integer. Got: ",
                          num_selection_heads);

    const auto& index_query_shape = input_shapes[0];
    const auto& block_index_keys_shape = input_shapes[1];
    const auto& key_length_shape = input_shapes[2];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           index_query_shape.rank().compatible(4),
                           "The shape of index_query must be 4D [N, Hi, L, Di].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           block_index_keys_shape.rank().compatible(3),
                           "The shape of block_index_keys must be 3D [N, B, Di].");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           key_length_shape.rank().compatible(1),
                           "The shape of key_length must be 1D [N] or [1].");

    TDim batch{};
    TDim num_queries{};
    if (index_query_shape.rank().is_static()) {
        batch = index_query_shape[0];
        num_queries = index_query_shape[2];
    }

    if (index_query_shape.rank().is_static() && block_index_keys_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(batch, batch, block_index_keys_shape[0]),
                               "The batch dimension of index_query and block_index_keys must be the same.");
        TDim merged;
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               TDim::merge(merged, index_query_shape[3], block_index_keys_shape[2]),
                               "The head size of index_query and block_index_keys must be the same.");
    }

    if (inputs_count == 4) {
        const auto& index_weights_shape = input_shapes[3];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               index_weights_shape.rank().compatible(3),
                               "The shape of index_weights must be 3D [N, L, Hi].");
        if (index_query_shape.rank().is_static() && index_weights_shape.rank().is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(batch, batch, index_weights_shape[0]),
                                   "The batch dimension of index_query and index_weights must be the same.");
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(num_queries, num_queries, index_weights_shape[1]),
                                   "The number of query tokens of index_query and index_weights must be the same.");
            TDim merged;
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(merged, index_query_shape[1], index_weights_shape[2]),
                                   "The number of index query heads of index_query and index_weights "
                                   "must be the same.");
        }
    }

    if (index_query_shape.rank().is_static() && index_query_shape[1].is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               index_query_shape[1].get_length() % num_selection_heads == 0,
                               "The number of index query heads must be divisible by num_selection_heads.");
    }

    const auto num_selected_blocks = token_budget / key_block_size;
    return {TRShape{batch, TDim(num_selection_heads), num_queries, TDim(num_selected_blocks)},
            TRShape{batch, TDim(num_selection_heads), num_queries}};
}
}  // namespace ov::op::v17

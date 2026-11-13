// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "utils.hpp"

namespace ov::op::internal {
/// \brief Infers the QSAIndexer output shapes; the op validates the input shapes.
///
/// block_index_keys_cache follows the QSAPrepare-17 rules: the input holds the blocks of the earlier calls, and the
/// output holds max(key_length) / key_block_size blocks. Without the key_length values, the output block count is
/// bounded by Bp + ceil(L / key_block_size).
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const QSAIndexer* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TShape::value_type;
    using In = QSAIndexer::InputIdx;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == In::COUNT);
    const auto& config = op->get_config();
    const auto key_block_size = static_cast<int64_t>(config.key_block_size);
    NODE_VALIDATION_CHECK(op, key_block_size > 0, "The attribute key_block_size must be positive.");

    const auto& index_query_shape = input_shapes[In::INDEX_QUERY];
    const auto& raw_index_keys_shape = input_shapes[In::RAW_INDEX_KEYS];
    const auto& cache_shape = input_shapes[In::BLOCK_INDEX_KEYS_CACHE];
    const auto is_query_rank_static = index_query_shape.rank().is_static();
    const auto is_raw_keys_rank_static = raw_index_keys_shape.rank().is_static();
    const auto is_cache_rank_static = cache_shape.rank().is_static();
    const auto batch = is_query_rank_static ? TDim(index_query_shape[0]) : TDim(ov::util::dim::inf_bound);
    const auto length = is_query_rank_static ? TDim(index_query_shape[1]) : TDim(ov::util::dim::inf_bound);
    const auto selection_heads = TDim(static_cast<int64_t>(config.num_selection_heads));

    auto num_blocks = TDim(ov::util::dim::inf_bound);
    if (const auto key_lengths = get_input_const_data_as<TRShape, int64_t>(op, In::KEY_LENGTH, ta)) {
        int64_t longest = 0;
        for (const auto& total : *key_lengths) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   total >= 0,
                                   "The key_length values must be non-negative. Got: ",
                                   total);
            const auto blocks = total / key_block_size;
            longest = blocks > longest ? blocks : longest;
        }
        num_blocks = TDim(longest);

        if (is_raw_keys_rank_static && raw_index_keys_shape[1].is_static() && is_cache_rank_static &&
            cache_shape[2].is_static()) {
            const auto new_tokens = raw_index_keys_shape[1].get_length();
            int64_t cached = 0;
            for (const auto& total : *key_lengths) {
                const auto past = total - new_tokens;
                const auto blocks = past > 0 ? past / key_block_size : 0;
                cached = blocks > cached ? blocks : cached;
            }
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   cache_shape[2].get_length() >= cached,
                                   "The third dimension of block_index_keys_cache must cover the complete blocks of "
                                   "the past tokens, at least: ",
                                   cached,
                                   ", got: ",
                                   cache_shape[2]);
        }
    } else if (is_raw_keys_rank_static && is_cache_rank_static) {
        // The past incomplete block contributes at most key_block_size - 1 tokens.
        const auto upper =
            cache_shape[2] +
            ov::util::dim::ceil_div(raw_index_keys_shape[1], static_cast<typename TDim::value_type>(key_block_size));
        if (upper.is_static()) {
            num_blocks = TDim(0, upper.get_length());
        }
    }

    auto cache_out =
        is_cache_rank_static
            ? TRShape{cache_shape[0], cache_shape[1], std::move(num_blocks), cache_shape[3]}
            : TRShape{batch,
                      is_raw_keys_rank_static ? TDim(raw_index_keys_shape[2]) : TDim(ov::util::dim::inf_bound),
                      std::move(num_blocks),
                      TDim(static_cast<int64_t>(config.indexer_head_dim))};
    return {TRShape{batch, selection_heads, length, TDim(static_cast<int64_t>(config.block_topk))},
            TRShape{batch, selection_heads, length},
            std::move(cache_out),
            input_shapes[In::PAST_INCOMPLETE_RAW_K],
            input_shapes[In::PAST_INCOMPLETE_ROPE]};
}
}  // namespace ov::op::internal
